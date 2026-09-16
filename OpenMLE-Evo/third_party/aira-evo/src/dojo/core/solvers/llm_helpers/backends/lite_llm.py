# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jsonschema
import litellm
import time
import httpx

from dataclasses_json import DataClassJsonMixin
from litellm import completion as completion_fn

litellm.api_version = "2024-12-01-preview"
litellm.set_verbose = False

NUM_RETRIES = 10
TIMEOUT = 1500
STREAM_ENV = "AIRA_LITELLM_STREAM"
TIMEOUT_ENV = "AIRA_LITELLM_TIMEOUT"
NUM_RETRIES_ENV = "AIRA_LITELLM_NUM_RETRIES"

# Context-window handling. When the requested output (`max_tokens`) plus the
# prompt would exceed the model's context window, the backend transparently
# shrinks `max_tokens` to fit rather than letting the whole task abort with a
# non-retryable ContextWindowExceededError.
#
# The prompt size is taken from the serving endpoint's own tokenizer when it
# exposes one (vLLM's `POST /tokenize`, which also reports `max_model_len`).
# Otherwise litellm's local estimate is used; for models litellm does not know
# that estimate comes from OpenAI's tiktoken and can undercount a Qwen/Llama
# prompt by a few percent, so an extra proportional margin is reserved on top of
# the fixed one.
CONTEXT_WINDOW_ENVS = ("AIRA_LITELLM_CONTEXT_WINDOW", "AIRA_LITELLM_MODEL_MAX_LEN")
TOKEN_MARGIN_ENV = "AIRA_LITELLM_TOKEN_MARGIN"  # fixed headroom (always reserved)
TOKEN_MARGIN_PCT_ENV = "AIRA_LITELLM_TOKEN_MARGIN_PCT"  # extra headroom when the count is estimated
MIN_OUTPUT_TOKENS_ENV = "AIRA_LITELLM_MIN_OUTPUT_TOKENS"  # never shrink output below this
CONTEXT_RETRIES_ENV = "AIRA_LITELLM_CONTEXT_RETRIES"  # reactive retries on window errors
REMOTE_TOKENIZE_ENV = "AIRA_LITELLM_REMOTE_TOKENIZE"  # ask the server for exact prompt counts
DEFAULT_TOKEN_MARGIN = 128
DEFAULT_TOKEN_MARGIN_PCT = 0.10
DEFAULT_MIN_OUTPUT_TOKENS = 512
DEFAULT_CONTEXT_RETRIES = 4
TOKENIZE_TIMEOUT = 15.0


# Configure logging
logger = logging.getLogger("Backend")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return {}


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: Dict[str, Any]  # JSON schema
    description: str

    def __post_init__(self):
        # Validate the JSON schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI's function format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_tool_choice_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic uses input_schema instead of parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Convert to Anthropic's tool choice format."""
        return {
            "type": "tool",  # Anthropic uses "tool" instead of "function"
            "name": self.name,
        }


class LiteLLMClient:
    PromptType = Union[str, Dict[str, Any], List[Any]]
    FunctionCallType = Dict[str, Any]
    OutputType = Union[str, FunctionCallType]

    def __init__(self, client_cfg):
        """
        Initialize the OpenAI client with any desired default arguments or configuration.
        """
        self.model = client_cfg.model_id
        self.base_url = client_cfg.base_url
        configured_api_key = str(getattr(client_cfg, "api_key", "") or "")
        if configured_api_key:
            self.api_key = configured_api_key
        else:
            api_key = os.getenv("PRIMARY_KEY_" + self.model.replace("-", "_").upper(), "")
            if api_key:
                self.api_key = api_key
            else:
                self.api_key = os.getenv("PRIMARY_KEY", "")
        self.use_azure_client = client_cfg.use_azure_client
        self.provider = client_cfg.provider
        if self.use_azure_client:
            self.model_prefix = "azure/"
        else:
            self.model_prefix = "openai/"

        # Name the server knows the model by (no litellm provider prefix).
        self.served_model = self.model
        self.model = self.model_prefix + self.model

        # Lazily discovered facts about the serving endpoint's tokenizer.
        self._tokenize_available: Optional[bool] = None
        self._server_max_model_len: Optional[int] = None

        logging.getLogger("httpx").setLevel(logging.WARNING)

    @property
    def client_content_key(self):
        return "content"

    def _calculate_cost(self, prompt_tokens, completion_tokens):
        """Calculate the API cost for a request based on token usage and provider-specific pricing."""
        cost = 0.0
        # Example cost calculation for different providers/models
        if self.provider.lower() == "openai":
            # Define cost per 1K tokens for some known OpenAI models (in USD)
            if "gpt-3.5" in self.model.lower():
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            elif "gpt-4" in self.model.lower():
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            else:
                # Default rates for other OpenAI models (if any)
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            # Calculate cost proportionally to the number of tokens (token counts are divided by 1000 for per-1K pricing)
            cost = (prompt_tokens / 1000.0) * prompt_cost_per_1k + (
                completion_tokens / 1000.0
            ) * completion_cost_per_1k
        elif self.provider.lower() == "anthropic":
            prompt_cost_per_1k = 0.0  # example cost per 1K tokens for prompts on Anthropic
            completion_cost_per_1k = 0.0  # example cost per 1K tokens for completions on Anthropic
            cost = (prompt_tokens / 1000.0) * prompt_cost_per_1k + (
                completion_tokens / 1000.0
            ) * completion_cost_per_1k
        else:
            # Other providers or default case
            # If costs are not known, leave as 0 or implement accordingly
            cost = 0.0
        return round(cost, 6)  # rounding to a reasonable number of decimal places for currency

    def count_tokens(self, text):
        """Utility method to count tokens in a given text string."""
        # In a real scenario, this should use the model's tokenizer for accuracy.
        # Here, we'll use a simple whitespace split as a placeholder.
        if text is None:
            return 0
        return len(text.split())

    def _should_stream(self, filtered_kwargs: dict[str, Any], func_spec: Optional[FunctionSpec]) -> bool:
        if func_spec is not None:
            return False
        if "stream" in filtered_kwargs:
            return bool(filtered_kwargs["stream"])
        base_url = str(self.base_url or "").lower()
        is_local_selfhosted = self.provider.lower() == "selfhosted" and (
            "localhost" in base_url or "127.0.0.1" in base_url
        )
        return _env_bool(STREAM_ENV, is_local_selfhosted)

    def _consume_stream(self, stream: Any) -> tuple[str, dict[str, Any]]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_stats: dict[str, Any] = {}
        finish_reason = None

        for chunk in stream:
            chunk_payload = _to_dict(chunk)
            usage = chunk_payload.get("usage")
            if usage:
                usage_stats.update(_to_dict(usage))

            for choice in chunk_payload.get("choices") or []:
                choice_payload = _to_dict(choice)
                delta = _to_dict(choice_payload.get("delta") or choice_payload.get("message"))

                content = delta.get("content")
                if content:
                    text_parts.append(str(content))

                reasoning = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("thinking")
                )
                if reasoning:
                    reasoning_parts.append(str(reasoning))

                if choice_payload.get("finish_reason") is not None:
                    finish_reason = choice_payload.get("finish_reason")

        if finish_reason is not None:
            usage_stats["finish_reason"] = finish_reason
        if reasoning_parts:
            usage_stats["reasoning_content"] = "".join(reasoning_parts)
        usage_stats["streamed"] = True

        return "".join(text_parts), usage_stats

    def _resolve_context_window(self) -> Optional[int]:
        """Best-effort lookup of the model's total context window (in tokens).

        Prefers an explicit env override, then the length the serving endpoint
        reported about itself (see `_count_prompt_tokens_remote`), then
        litellm's registry. Returns None when the window cannot be determined.
        """
        for env_name in CONTEXT_WINDOW_ENVS:
            raw = os.getenv(env_name)
            if raw:
                try:
                    return int(raw)
                except ValueError:
                    logger.warning("Ignoring non-integer %s=%r", env_name, raw)
        if self._server_max_model_len:
            return self._server_max_model_len
        try:
            info = litellm.get_model_info(self.model) or {}
        except Exception:
            info = {}
        for key in ("max_input_tokens", "max_tokens"):
            value = info.get(key)
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return None

    def _tokenize_url(self) -> Optional[str]:
        """`/tokenize` endpoint next to the OpenAI-compatible `/v1` base URL."""
        base = str(self.base_url or "").strip().rstrip("/")
        if not base:
            return None
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base + "/tokenize"

    def _count_prompt_tokens_remote(
        self,
        messages: List[Dict[str, str]],
        func_spec: Optional[FunctionSpec],
        chat_template_kwargs: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """Exact prompt size from the server's own tokenizer, or None.

        Targets vLLM's `POST /tokenize`, which applies the served chat template
        and returns both the token count and the model's `max_model_len`. The
        endpoint is probed once; if it is missing the backend stops asking.
        """
        if self.use_azure_client or not _env_bool(REMOTE_TOKENIZE_ENV, True):
            return None
        if self._tokenize_available is False:
            return None
        url = self._tokenize_url()
        if not url:
            return None

        payload: Dict[str, Any] = {
            "model": self.served_model,
            "messages": messages,
            "add_generation_prompt": True,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if func_spec is not None:
            payload["tools"] = [{"type": "function", "function": func_spec.as_openai_tool_dict}]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=TOKENIZE_TIMEOUT)
        except Exception as e:
            logger.debug("Remote tokenize failed (%s); using local estimate.", e)
            return None

        if 400 <= response.status_code < 500 and response.status_code != 429:
            # Not a vLLM-style server (404/405) or it rejects the request
            # shape; a local estimate is used from now on.
            self._tokenize_available = False
            logger.info(
                "Tokenize endpoint %s unavailable (HTTP %s); using local token estimates.",
                url,
                response.status_code,
            )
            return None
        if response.status_code >= 300:
            return None

        try:
            body = response.json()
        except ValueError:
            return None
        count = body.get("count") if isinstance(body, dict) else None
        if not isinstance(count, int) or count < 0:
            return None
        max_model_len = body.get("max_model_len")
        if isinstance(max_model_len, int) and max_model_len > 0:
            self._server_max_model_len = max_model_len
        self._tokenize_available = True
        return count

    def _estimate_prompt_tokens(
        self, messages: List[Dict[str, str]], func_spec: Optional[FunctionSpec]
    ) -> Optional[int]:
        """Approximate the prompt token count (tools included when possible)."""
        tools = [func_spec.as_openai_tool_dict] if func_spec is not None else None
        for kwargs in ({"tools": tools}, {}):
            try:
                return int(
                    litellm.token_counter(model=self.model, messages=messages, **kwargs)
                )
            except Exception:
                continue
        return None

    def _count_prompt_tokens(
        self,
        messages: List[Dict[str, str]],
        func_spec: Optional[FunctionSpec],
        chat_template_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[int], bool]:
        """Return (prompt_tokens, exact). `exact` is False for local estimates."""
        count = self._count_prompt_tokens_remote(messages, func_spec, chat_template_kwargs)
        if count is not None:
            return count, True
        return self._estimate_prompt_tokens(messages, func_spec), False

    @staticmethod
    def _token_margin(prompt_tokens: int, exact: bool) -> int:
        """Headroom to reserve above the prompt count.

        An exact server-side count only needs the fixed margin. A local
        estimate may come from a different tokenizer than the one serving the
        model, so a proportional share of the prompt is reserved as well.
        """
        margin = _env_int(TOKEN_MARGIN_ENV, DEFAULT_TOKEN_MARGIN)
        if not exact:
            pct = _env_float(TOKEN_MARGIN_PCT_ENV, DEFAULT_TOKEN_MARGIN_PCT)
            margin += int(math.ceil(prompt_tokens * max(pct, 0.0)))
        return margin

    def _cap_max_tokens(
        self,
        filtered_kwargs: Dict[str, Any],
        messages: List[Dict[str, str]],
        func_spec: Optional[FunctionSpec],
    ) -> None:
        """Proactively shrink `max_tokens` so prompt + output fits the window."""
        requested = filtered_kwargs.get("max_tokens")
        if not requested:
            return
        extra_body = filtered_kwargs.get("extra_body")
        chat_template_kwargs = (
            extra_body.get("chat_template_kwargs") if isinstance(extra_body, dict) else None
        )
        # Count first: a server-side count may also reveal the context window.
        prompt_tokens, exact = self._count_prompt_tokens(messages, func_spec, chat_template_kwargs)
        window = self._resolve_context_window()
        if not window or prompt_tokens is None:
            return
        margin = self._token_margin(prompt_tokens, exact)
        floor = _env_int(MIN_OUTPUT_TOKENS_ENV, DEFAULT_MIN_OUTPUT_TOKENS)
        budget = window - prompt_tokens - margin
        if budget >= requested:
            return
        new_max = max(floor, budget)
        if new_max >= requested:
            return
        logger.warning(
            "Capping max_tokens %s -> %s to fit context window %s "
            "(%s%s prompt tokens, margin %s).",
            requested,
            new_max,
            window,
            "" if exact else "~",
            prompt_tokens,
            margin,
        )
        filtered_kwargs["max_tokens"] = new_max

    @staticmethod
    def _parse_context_error(message: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract (context_window, prompt_tokens) from a provider error string."""
        window = None
        prompt_tokens = None
        m = re.search(r"maximum context length is (\d+)", message)
        if m:
            window = int(m.group(1))
        m = re.search(r"prompt contains at least (\d+)", message)
        if not m:
            # Alternate OpenAI/vLLM phrasings.
            m = re.search(r"(?:messages? resulted in|you requested)\D*(\d+)\s+(?:input )?tokens", message)
        if m:
            prompt_tokens = int(m.group(1))
        return window, prompt_tokens

    def _shrink_max_tokens_from_error(
        self, message: str, current_max: Optional[int]
    ) -> Optional[int]:
        """Compute a max_tokens that fits, using counts the server reported.

        Returns None when the error is unparseable, when even the minimum
        output would not fit (prompt itself too large), or when shrinking would
        not actually reduce the current value.
        """
        window, prompt_tokens = self._parse_context_error(message)
        if not window or not prompt_tokens:
            return None
        margin = _env_int(TOKEN_MARGIN_ENV, DEFAULT_TOKEN_MARGIN)
        floor = _env_int(MIN_OUTPUT_TOKENS_ENV, DEFAULT_MIN_OUTPUT_TOKENS)
        budget = window - prompt_tokens - margin
        if budget < floor:
            return None
        if current_max is not None and budget >= current_max:
            return None
        return budget

    def _query_client(
        self,
        messages: List[Dict[str, str]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Tuple[OutputType, Dict[str, Any]]:
        # Prepare function specifications if provided
        func_spec = None
        if json_schema and function_name and function_description:
            func_spec = FunctionSpec(function_name, json.loads(json_schema), function_description)

        # Always include necessary model parameters
        model_kwargs = dict(model_kwargs or {})
        model_kwargs["model"] = self.model
        model_kwargs["base_url"] = self.base_url
        model_kwargs["api_key"] = self.api_key
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        # Attach function specifications if provided
        if func_spec is not None:
            filtered_kwargs["functions"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["function_call"] = {"name": func_spec.name}

        if "minimax" in self.model.lower() and not any(
            message.get("role") == "user" for message in messages
        ):
            # MiniMax rejects system-only chats, which AIRA operators use by design.
            merged_content = "\n\n".join(
                str(message.get(self.client_content_key, "")) for message in messages
            ).strip()
            messages = [{"role": "user", self.client_content_key: merged_content}]

        use_stream = self._should_stream(filtered_kwargs, func_spec)
        include_usage = bool(filtered_kwargs.pop("include_usage", True))
        if use_stream:
            filtered_kwargs["stream"] = True
            if include_usage:
                filtered_kwargs.setdefault("stream_options", {"include_usage": True})
        else:
            filtered_kwargs.pop("stream", None)
            filtered_kwargs.pop("stream_options", None)

        num_retries = _env_int(NUM_RETRIES_ENV, NUM_RETRIES)
        request_timeout = _env_float(TIMEOUT_ENV, TIMEOUT)
        # `max_retries` is handled by the OpenAI SDK client, which retries
        # connection errors, timeouts, 408/409/429 and 5xx with backoff and
        # never a 400. litellm's own `num_retries` is deliberately NOT set: it
        # re-sends every `openai.APIError` verbatim, so a context-window 400
        # would be replayed `num_retries` times before the handler below could
        # shrink `max_tokens`.
        filtered_kwargs["max_retries"] = num_retries
        filtered_kwargs["request_timeout"] = httpx.Timeout(timeout=request_timeout)

        # Proactively fit the request inside the model's context window so the
        # prompt plus requested output never overflows it.
        self._cap_max_tokens(filtered_kwargs, messages, func_spec)

        # Record start time for latency measurement
        start_time = time.monotonic()

        # Execute the LLM call, with fallback for function calling errors
        completion = None
        output = None
        usage_stats: dict[str, Any] = {}

        def _run_completion() -> tuple[Any, Any, dict[str, Any]]:
            comp = completion_fn(messages=messages, **filtered_kwargs)
            out = None
            stats: dict[str, Any] = {}
            if use_stream:
                out, stats = self._consume_stream(comp)
            return comp, out, stats

        max_ctx_retries = _env_int(CONTEXT_RETRIES_ENV, DEFAULT_CONTEXT_RETRIES)
        ctx_attempts = 0
        while True:
            try:
                completion, output, usage_stats = _run_completion()
                break
            except litellm.ContextWindowExceededError as e:
                # Reactive fit: the server reported the exact prompt/window
                # sizes, so shrink max_tokens to match rather than aborting.
                new_max = self._shrink_max_tokens_from_error(
                    str(e), filtered_kwargs.get("max_tokens")
                )
                ctx_attempts += 1
                if new_max is None or ctx_attempts > max_ctx_retries:
                    raise
                logger.warning(
                    "Context window exceeded; retrying with reduced "
                    "max_tokens=%s (attempt %s/%s).",
                    new_max,
                    ctx_attempts,
                    max_ctx_retries,
                )
                filtered_kwargs["max_tokens"] = new_max
                continue
            except litellm.BadRequestError as e:
                if "function calling" in str(e).lower() or "functions" in str(e).lower():
                    logger.warning(
                        "Function calling was attempted but is not supported by this model. "
                        "Falling back to plain text generation."
                    )
                    # Remove function calling parameters and retry
                    filtered_kwargs.pop("functions", None)
                    filtered_kwargs.pop("function_call", None)
                    use_stream = self._should_stream(filtered_kwargs, None)
                    if use_stream:
                        filtered_kwargs["stream"] = True
                        if include_usage:
                            filtered_kwargs.setdefault("stream_options", {"include_usage": True})
                    completion, output, usage_stats = _run_completion()
                    break
                else:
                    raise

        # Calculate latency
        latency = time.monotonic() - start_time

        # Extract usage stats from the LLM response (if available)
        choice = None
        reasoning_content = None
        if not use_stream and completion is not None:
            choice = completion.choices[0]
            usage_stats = completion.to_dict().get("usage", {}) or {}
            message = choice.message
            reasoning_content = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or getattr(message, "thinking", None)
            )
            if reasoning_content is None:
                provider_specific_fields = getattr(message, "provider_specific_fields", None)
                provider_specific_fields = _to_dict(provider_specific_fields)
                reasoning_content = (
                    provider_specific_fields.get("reasoning_content")
                    or provider_specific_fields.get("reasoning")
                    or provider_specific_fields.get("thinking")
                )
                reasoning_details = provider_specific_fields.get("reasoning_details")
                if reasoning_content is None and isinstance(reasoning_details, list):
                    reasoning_parts = [
                        str(detail.get("text"))
                        for detail in reasoning_details
                        if isinstance(detail, dict) and detail.get("text")
                    ]
                    if reasoning_parts:
                        reasoning_content = "".join(reasoning_parts)
            if reasoning_content:
                usage_stats["reasoning_content"] = str(reasoning_content)

        # Add latency and success status to the stats
        usage_stats["latency"] = latency
        usage_stats["success"] = True

        # If token counts are not available from the response, estimate them.
        if "prompt_tokens" not in usage_stats:
            prompt_text = " ".join([str(m.get(self.client_content_key, "")) for m in messages])
            usage_stats["prompt_tokens"] = self.count_tokens(prompt_text)
        if "completion_tokens" not in usage_stats:
            completion_text = output if isinstance(output, str) else choice.message.content
            usage_stats["completion_tokens"] = self.count_tokens(completion_text)
        usage_stats["total_tokens"] = usage_stats["prompt_tokens"] + usage_stats["completion_tokens"]

        # Calculate cost using a helper (this method can adjust for different backends)
        usage_stats["cost"] = self._calculate_cost(usage_stats["prompt_tokens"], usage_stats["completion_tokens"])

        # Parse the response as before
        if use_stream:
            output = output or ""
        elif func_spec is None or "functions" not in filtered_kwargs:
            output = choice.message.content or reasoning_content or ""
        else:
            try:
                function_call = choice.message.function_call
            except:
                function_call = None
            if not function_call:
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {choice.message.content}"
                )
                output = choice.message.content or reasoning_content or ""
            else:
                if not str(function_call.name).strip() == str(func_spec.name).strip():
                    logger.warning(
                        f"Function name mismatch: expected {func_spec.name}, "
                        f"got {function_call.name}. Fallback to text."
                    )
                    output = choice.message.content or reasoning_content or ""
                else:
                    try:
                        output = json.loads(function_call.arguments)
                    except json.JSONDecodeError as ex:
                        logger.error(f"Error decoding function arguments:\n{function_call.arguments}")
                        raise ex

        return output, usage_stats

    def query(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
        **model_kwargs,
    ) -> OutputType:
        """
        General LLM query for various backends with a single system and user message.
        Supports function calling for some backends.

        Args:
            system_message (PromptType | None): Uncompiled system message.
            user_message (PromptType | None): Uncompiled user message.
            model (str): Identifier for the model to use (e.g., "gpt-4-turbo").
            temperature (float | None, optional): Sampling temperature.
            max_tokens (int | None, optional): Maximum number of tokens to generate.
            func_spec (FunctionSpec | None, optional): Optional FunctionSpec for function calling.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            OutputType: A string completion or a dict with function call details.
        """

        if self.model == "azure/o1-preview" or self.model == "azure/o3-mini":
            messages = [{"role": "user", self.client_content_key: m[self.client_content_key]} for m in messages]
            if "temperature" in model_kwargs:
                model_kwargs.pop("temperature")

        output, usage_stats = self._query_client(
            messages=messages,
            model_kwargs=model_kwargs,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )

        return output, usage_stats

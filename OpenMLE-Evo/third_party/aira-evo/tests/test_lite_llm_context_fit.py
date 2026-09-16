"""Regression tests for context-window fitting in the litellm backend.

Scenario from a real run: vLLM served frontis-ma1 with max_model_len=40960,
the runner asked for max_tokens=32768, litellm's tiktoken-based estimate said
~8798 prompt tokens but the server counted 8927. Capping to 32034 (fixed margin
128) overshot the window by one token, vLLM answered 400, and litellm replayed
the identical request `num_retries` times before the backend could react.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import litellm
import pytest

os.environ.setdefault("LOGGING_DIR", "/tmp")  # dojo config import requires it

from dojo.core.solvers.llm_helpers.backends import lite_llm  # noqa: E402
from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient  # noqa: E402

WINDOW = 40960
REQUESTED = 32768
SERVER_PROMPT_TOKENS = 8927
LOCAL_ESTIMATE = 8798
MESSAGES = [
    {"role": "system", "content": "You are improving a Python solution."},
    {"role": "user", "content": "Return only Python code."},
]


def make_client(base_url: str = "http://127.0.0.1:8000/v1") -> LiteLLMClient:
    cfg = SimpleNamespace(
        model_id="frontis-ma1",
        base_url=base_url,
        api_key="EMPTY",
        use_azure_client=False,
        provider="selfhosted",
    )
    return LiteLLMClient(cfg)


class FakeResponse:
    def __init__(self, status_code: int, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _no_local_estimate(**kwargs):
    raise RuntimeError("no tokenizer available")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        *lite_llm.CONTEXT_WINDOW_ENVS,
        lite_llm.TOKEN_MARGIN_ENV,
        lite_llm.TOKEN_MARGIN_PCT_ENV,
        lite_llm.MIN_OUTPUT_TOKENS_ENV,
        lite_llm.REMOTE_TOKENIZE_ENV,
        lite_llm.NUM_RETRIES_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_remote_tokenize_request_shape_and_exact_cap(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(200, {"count": SERVER_PROMPT_TOKENS, "max_model_len": WINDOW})

    monkeypatch.setattr(lite_llm.httpx, "post", fake_post)
    client = make_client()
    kwargs = {
        "max_tokens": REQUESTED,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }

    client._cap_max_tokens(kwargs, MESSAGES, None)

    assert len(calls) == 1
    assert calls[0]["url"] == "http://127.0.0.1:8000/tokenize"
    assert calls[0]["json"]["model"] == "frontis-ma1"  # served name, no "openai/" prefix
    assert calls[0]["json"]["messages"] == MESSAGES
    assert calls[0]["json"]["add_generation_prompt"] is True
    assert calls[0]["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert calls[0]["headers"] == {"Authorization": "Bearer EMPTY"}
    # Exact count: only the fixed margin is reserved; the window came from the server.
    assert kwargs["max_tokens"] == WINDOW - SERVER_PROMPT_TOKENS - lite_llm.DEFAULT_TOKEN_MARGIN
    assert SERVER_PROMPT_TOKENS + kwargs["max_tokens"] <= WINDOW


def test_local_estimate_reserves_proportional_margin(monkeypatch):
    # Plain OpenAI-compatible server without /tokenize.
    monkeypatch.setattr(lite_llm.httpx, "post", lambda *a, **k: FakeResponse(404))
    monkeypatch.setattr(lite_llm.litellm, "token_counter", lambda **k: LOCAL_ESTIMATE)
    monkeypatch.setenv(lite_llm.CONTEXT_WINDOW_ENVS[0], str(WINDOW))
    client = make_client()
    kwargs = {"max_tokens": REQUESTED}

    client._cap_max_tokens(kwargs, MESSAGES, None)

    # The old fixed margin let 8927 + 32034 = 40961 through; the server's real
    # count must now fit with room to spare.
    assert SERVER_PROMPT_TOKENS + kwargs["max_tokens"] <= WINDOW
    expected_margin = lite_llm.DEFAULT_TOKEN_MARGIN + math.ceil(
        LOCAL_ESTIMATE * lite_llm.DEFAULT_TOKEN_MARGIN_PCT
    )
    assert kwargs["max_tokens"] == WINDOW - LOCAL_ESTIMATE - expected_margin
    assert client._tokenize_available is False


def test_missing_tokenize_endpoint_is_probed_once(monkeypatch):
    posts = []

    def fake_post(*a, **k):
        posts.append(1)
        return FakeResponse(404)

    monkeypatch.setattr(lite_llm.httpx, "post", fake_post)
    monkeypatch.setattr(lite_llm.litellm, "token_counter", lambda **k: LOCAL_ESTIMATE)
    monkeypatch.setenv(lite_llm.CONTEXT_WINDOW_ENVS[0], str(WINDOW))
    client = make_client()

    for _ in range(3):
        client._cap_max_tokens({"max_tokens": REQUESTED}, MESSAGES, None)

    assert len(posts) == 1


def test_remote_tokenize_can_be_disabled(monkeypatch):
    monkeypatch.setenv(lite_llm.REMOTE_TOKENIZE_ENV, "0")
    monkeypatch.setattr(
        lite_llm.httpx, "post", lambda *a, **k: pytest.fail("server must not be called")
    )
    monkeypatch.setattr(lite_llm.litellm, "token_counter", lambda **k: LOCAL_ESTIMATE)
    monkeypatch.setenv(lite_llm.CONTEXT_WINDOW_ENVS[0], str(WINDOW))
    client = make_client()
    kwargs = {"max_tokens": REQUESTED}

    client._cap_max_tokens(kwargs, MESSAGES, None)

    assert kwargs["max_tokens"] < REQUESTED


def test_context_error_reaches_handler_first_try_and_is_shrunk(monkeypatch):
    """A 400 must reach the backend's own handler on the first attempt (litellm
    `num_retries` unset), which retries once with a max_tokens fitted from the
    counts the server reported."""
    monkeypatch.setattr(lite_llm.httpx, "post", lambda *a, **k: FakeResponse(404))
    monkeypatch.setattr(lite_llm.litellm, "token_counter", _no_local_estimate)

    seen = []
    error_message = (
        f"This model's maximum context length is {WINDOW} tokens. However, you requested "
        f"32034 output tokens and your prompt contains at least {SERVER_PROMPT_TOKENS} "
        "input tokens, for a total of at least 40961 tokens."
    )

    def fake_completion(messages, **kwargs):
        seen.append(dict(kwargs))
        if len(seen) == 1:
            raise litellm.ContextWindowExceededError(
                message=error_message, model="frontis-ma1", llm_provider="openai"
            )
        return litellm.ModelResponse(
            choices=[
                {"message": {"role": "assistant", "content": "print('ok')"}, "finish_reason": "stop"}
            ]
        )

    monkeypatch.setattr(lite_llm, "completion_fn", fake_completion)
    client = make_client()

    output, stats = client.query(MESSAGES, max_tokens=REQUESTED, stream=False)

    assert output == "print('ok')"
    assert stats["success"] is True
    assert len(seen) == 2
    assert "num_retries" not in seen[0]
    assert seen[0]["max_retries"] == lite_llm.NUM_RETRIES
    assert seen[0]["max_tokens"] == REQUESTED
    assert seen[1]["max_tokens"] == WINDOW - SERVER_PROMPT_TOKENS - lite_llm.DEFAULT_TOKEN_MARGIN

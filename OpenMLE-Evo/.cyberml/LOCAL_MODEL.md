# Running the cyber-ML loop against a self-hosted model (Frontis-MA1-35B)

Point the OpenMLE-Evo search loop at a **local / rented-GPU** copy of
[`FrontisAI/Frontis-MA1-35B`](https://huggingface.co/FrontisAI/Frontis-MA1-35B)
instead of a hosted API model.

## (a) TL;DR

**No runner changes are needed.** `scripts/run_naturebench_local.py` already
speaks to any OpenAI-compatible `/v1` endpoint. You only have to:

1. **Serve** Frontis-MA1-35B behind an OpenAI-compatible server (vLLM/SGLang) on
   a GPU box — see (c), or use `scripts/serve_frontis_ma1.sh`.
2. **Point the runner at it**, two ways (see (d)):
   - **SSH tunnel** to a rented remote GPU (`--model-ssh-host/--model-ssh-port`) —
     recommended for cloud GPUs; the runner auto-uses `PRIMARY_KEY=EMPTY`.
   - **Direct URL** (`--model-base-url http://host:8000/v1`) for a LAN / already
     reachable endpoint.

The candidate ML code still runs locally in the repo `.venv` — the model endpoint
is orthogonal to where candidates execute.

Frontis-MA1-35B is a Qwen3.6-35B-A3B **MoE** (35B total, ~3B activated per token),
so decode throughput is high and cheap — well suited to the island evo loop's many
concurrent calls. Native context is 262,144, but post-training SFT cut off at
32,768; serve with `--max-model-len 32768` (the trained ceiling, and it comfortably
fits the operator prompts, which run ~25k prompt tokens in practice).

## (b) GPU sizing (35B total weights)

| Precision   | Approx VRAM | Example GPU                                   | Notes |
|-------------|-------------|-----------------------------------------------|-------|
| BF16/FP16   | ~70 GB      | 1×H100-80GB                                    | Cleanest single-card fit. |
| BF16/FP16   | ~70 GB      | 1×A100-80GB                                     | Tight — lower `--max-model-len` and/or `--gpu-memory-utilization`. |
| BF16/FP16   | ~70 GB      | 2×A100/L40S-40–48GB, `--tensor-parallel-size 2` | Split across two cards. |
| FP8 / AWQ   | ~35 GB      | 1×A100-40GB or 1×L40S-48GB                       | Quantized weights; good throughput. |
| GGUF Q4_K_M | ~20 GB      | 1×24GB card, or Apple M4 Pro (llama.cpp/Ollama) | Use `FrontisAI/Frontis-MA1-35B-GGUF`. |

Because only ~3B params are active per token, a single H100 already sustains the
loop's concurrency comfortably.

## (c) Serve it

### vLLM (recommended)

Install once (`pip install vllm`), then on the GPU box:

```bash
vllm serve FrontisAI/Frontis-MA1-35B \
  --served-model-name frontis-ma1 \
  --port 8000 \
  --tensor-parallel-size 1 \        # set to the number of GPUs
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --language-model-only
```

Or just run the bundled helper (same command, env-configurable):

```bash
TP=1 PORT=8000 scripts/serve_frontis_ma1.sh
```

`--enable-prefix-caching` matters here: the Draft/Improve/Debug/Crossover operator
prompts share large common prefixes, so prefix caching cuts prompt-token work
substantially.

### SGLang (alternative)

```bash
python -m sglang.launch_server --model-path FrontisAI/Frontis-MA1-35B \
  --served-model-name frontis-ma1 --port 8000 --tp 1 --context-length 32768
```

### GGUF / local (no dedicated GPU)

Serve the quantized `FrontisAI/Frontis-MA1-35B-GGUF` (Q4_K_M, ~20 GB) with an
OpenAI-compatible server:

```bash
# llama.cpp
llama-server -hf FrontisAI/Frontis-MA1-35B-GGUF --port 8000 --ctx-size 32768 --alias frontis-ma1
# or Ollama, then point the runner at http://127.0.0.1:11434/v1
```

## (d) Wire it into the loop

`--model-id` **must equal the server's served-model-name** (`frontis-ma1`).

### Rented remote GPU via SSH tunnel (recommended for cloud GPUs)

The runner opens the tunnel, probes `/v1/models`, and **auto-sets
`PRIMARY_KEY=EMPTY`** for the no-auth local server. `--model-ssh-port` is the API
port on the remote box; `--model-ssh-remote-host` (default `127.0.0.1`) is where
vLLM listens on that box.

```bash
# cwd: OpenMLE-Evo
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ../NatureBench \
  --local-python .venv/bin/python \
  --data-dir .cyberml/data --skip-download \
  --task nsl-kdd-nids \
  --model-ssh-host user@gpu-box.example.com \
  --model-ssh-port 8000 \
  --model-id frontis-ma1 \
  -- llm_concurrency=6 search.runner.solver.step_limit=40
```

### Direct URL (LAN / reverse-proxied / already tunnelled)

Put `PRIMARY_KEY=EMPTY` in `.env` (the runner auto-loads `.env`; if the endpoint
requires a real key, set `PRIMARY_KEY` to it instead). `--model-id` is
authoritative even if `/v1/models` can't be listed.

```bash
# cwd: OpenMLE-Evo   (.env contains: PRIMARY_KEY=EMPTY)
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ../NatureBench \
  --local-python .venv/bin/python \
  --data-dir .cyberml/data --skip-download \
  --task nsl-kdd-nids \
  --model-base-url http://gpu-box.example.com:8000/v1 \
  --model-id frontis-ma1 \
  -- llm_concurrency=6 search.runner.solver.step_limit=40
```

Swap `--task nsl-kdd-nids` for any cyber-ML task id, or drop `--task` to run the
whole task set. The candidate runtime stays local via `--local-python` +
`--data-dir .cyberml/data`.

## (e) Reasoning / thinking traces

Frontis-MA1 emits chain-of-thought. With `--reasoning-parser qwen3`, vLLM places
that in a separate `reasoning_content` field and leaves the normal `content` with
the final answer, so the evo operators that parse ```` ```code``` ```` blocks work
unchanged — no prompt or parser edits required.

## (f) Concurrency & throughput tuning

- The loop's LLM concurrency is the Hydra `llm_concurrency=<N>` override (passed
  after `--`). Start at `llm_concurrency=4–8` for a single H100 and tune to what
  the server batches without latency blowing up; vLLM does continuous batching, so
  a modest N keeps the GPU busy.
- Keep `--enable-prefix-caching` on — operator prompts share long prefixes.
- If you see OOM at load, lower `--gpu-memory-utilization` (e.g. 0.85) or
  `--max-model-len`; if you see it mid-run, lower `llm_concurrency`.

## (g) License

Frontis-MA1-35B is **CC BY-NC 4.0 (non-commercial)**. Research and benchmarking —
exactly this self-improvement study — are fine. Do not use its outputs for a
commercial product.

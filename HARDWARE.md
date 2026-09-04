# Hardware requirements

What machine you need depends entirely on **which** experiment you run. This repo
spans four very different workloads, from a CPU-only unit test to an 8×H200
training job. Pick your tier below.

Two conventions in this document:

- **Stated** = written down in this repo or the Frontis-MA1 paper. Cited.
- **Estimate** = my sizing arithmetic or standard practice, not from the repo.
  Treat as a starting point, verify on your own hardware.

A quick orientation:

| Tier | Workload | GPU | Typical VM |
| --- | --- | --- | --- |
| 0 | `ctf_gym` tests + benchmark containers | **none** | 2 vCPU / 4 GB / 20 GB |
| 1 | One NatureBench task, hosted API model | **none** | 8 vCPU / 32 GB / 100 GB |
| 2 | Self-host Frontis-MA1 for inference | 1×H200, or 2×80 GB | 32 vCPU / 128 GB / 300 GB |
| 3 | MLE-Bench Lite reproduction | 1×RTX 4090 (12 GB cap) per sandbox worker | see §3 |
| 4 | SFT or RL training | **8×H200** | see §4 |

**The important takeaway: tiers 0 and 1 need no GPU at all.** You can do real work
— the whole CTF gym, and a complete NatureBench search trajectory — on a plain
CPU VM, as long as you point the model at a hosted API endpoint.

---

## Tier 0 — `ctf_gym` only (no GPU)

The CTF gym is pure CPU. `ctf_gym/tests/test_ctf_gym.py` starts a local uvicorn
verifier on port 18556 and runs two tasks; the benchmark targets
(`ctf_gym/benchmarks/*/Dockerfile`) are `ubuntu:24.04` + `python3`, nothing more.

| Resource | Minimum | Notes |
| --- | --- | --- |
| GPU | none | |
| vCPU | 2 | |
| RAM | 4 GB | |
| Disk | 20 GB | ~2 GB of that is the uv environment |
| Other | Docker (only to build the benchmark targets; the unit tests don't need it) | |

Any small cloud instance works — `t3.medium`, `e2-medium`, a $6/mo VPS.

```bash
uv sync && make test        # Ran 2 tests ... OK
```

---

## Tier 1 — One NatureBench task, hosted model (no GPU)

This is the smallest **real** experiment: a full OpenMLE-Evo search loop with
experience memory, three-factor parent selection, and `aggregate_improvement`
fitness, on task `s42256-023-00611-x`. See
[`OpenMLE-Evo/benchmarks/naturebench_local_quick/README.md`](OpenMLE-Evo/benchmarks/naturebench_local_quick/README.md).

**Stated:** that task "runs reliably on CPU and Apple Silicon". No GPU is used
anywhere in this path — the model comes from an OpenAI-compatible HTTP endpoint,
and generated candidate code executes locally in a Conda environment.

| Resource | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| GPU | none | none | |
| vCPU | 4 | 8 | Candidate code is single-cell analysis (scanpy, umap-learn, xgboost) |
| RAM | 16 GB | 32 GB | *Estimate* — driven by the candidate programs, not the search loop |
| Disk | 60 GB | 100 GB | uv env + Conda env + NatureBench repo + run outputs |
| Network | — | — | Outbound HTTPS to your model provider |

**Stated budgets:** `--smoke` caps the run at a **1800 s (30 min)** model+sandbox
budget and one generated candidate. The default (non-smoke) single-task run keeps
the formal experiment limits: **4 h effective budget, 6 h wall-clock, ≤160 nodes**.

### ⚠️ Security boundary

**Stated, and worth repeating:** Conda isolates Python dependencies — *not* files,
networking, processes, or host permissions. Model-generated code runs **with your
user's full access**. Use a disposable VM for this, never your workstation or
anything holding credentials. Formal runs should use the Docker or SCM execution
profiles instead.

This is a good reason to run tier 1 on a throwaway cloud VM rather than locally.

### Cost note

At this tier your spend is API tokens, not compute. A CPU VM plus a hosted
endpoint is dramatically cheaper than renting a GPU box, and the search behaviour
is identical.

---

## Tier 2 — Self-hosting Frontis-MA1

Only needed if you want to run the released weights instead of a hosted API.

Both models are **MoE** — the SFT launchers name them `Qwen3-30B-A3B` and
`Qwen3.5/3.6-35B-A3B` (`OpenMLE-ERL/SFT/slime_scripts/*/train.sh`). `A3B` means
~3B active parameters per token: throughput behaves like a small model, but
**VRAM is set by the total parameter count**, so you still need to fit ~30–35B
weights.

**Estimates** (params × 2 bytes for BF16, plus KV cache and runtime overhead):

| Serving mode | Weights | Workable GPUs | SGLang flag |
| --- | --- | --- | --- |
| BF16, 30B | ~60 GB | 1×H200 (141 GB) · 2×H100/A100 80 GB · 4×L40S/A6000 48 GB | `--tp-size 2` / `4` |
| BF16, 35B | ~70 GB | 1×H200 · 2×H100/A100 80 GB · 4×L40S/A6000 48 GB | `--tp-size 2` / `4` |
| GGUF Q4, 35B | ~18–20 GB | 1×24 GB card (RTX 4090 / L4 / A10G) | n/a — see below |

Add headroom for the KV cache; the training context is 32,768 tokens, and serving
at that length on a 24 GB card leaves little room.

Launch command (**stated**, from the local-quick README):

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /absolute/path/to/model \
  --served-model-name my-served-model \
  --host 127.0.0.1 --port 30010 --tp-size 2
```

> The documented SGLang path expects the **BF16 HF checkpoint**. The
> [GGUF derivatives](https://huggingface.co/collections/FrontisAI/frontis-ma1)
> are the cheap route — serve them with llama.cpp or Ollama and point
> `--model-base-url` at that server's OpenAI-compatible `/v1`. OpenMLE-Evo only
> ever speaks the OpenAI protocol, so any compliant server works.

Disk: **~150 GB estimate** for BF16 weights plus the Hugging Face cache; ~40 GB
for GGUF.

The search process does not need to sit on the GPU box. `run_naturebench_local.py`
supports `--model-ssh-host` / `--model-ssh-port`, which tunnels to a remote
loopback-only SGLang — candidate code, data, and evaluation stay local and only
prompts cross the tunnel.

---

## Tier 3 — MLE-Bench Lite reproduction

This is the paper's headline result, and it is a substantial commitment.

**Stated (README, abstract):** MLE-Bench Lite under a **12-hour per-task budget on
one RTX 4090 capped at 12 GB VRAM**.

Read that carefully — the 4090 is the **execution sandbox** GPU, the machine that
runs the model-generated ML code. It is *separate* from whatever serves the model.
The 12 GB cap is deliberate: it constrains what candidate programs can do.

| Piece | Requirement |
| --- | --- |
| Model service | Tier 2, or a hosted API |
| Sandbox worker | 1× RTX 4090-class GPU, capped at 12 GB VRAM |
| Search controller | Tier 1 CPU VM — the loop is single-process async and I/O-bound |
| Data | MLE-Bench task data via `mle-bench` + a Kaggle API token |

**Time:** MLE-Bench Lite is 22 tasks × 12 h = **264 GPU-hours** per sweep on a
single sandbox worker. The async profile (`AIRAEVO_WORKERS=8`,
`./scripts/run_multi_gpu.sh`) runs 8 concurrent workers — that wants **8 sandbox
GPUs plus a sandbox router**, and is what "OpenMLE-Evo-Max" refers to.

**Disk:** the repo does not state a figure. Kaggle competition data for the Lite
split is large; **budget several hundred GB** and check per task. Setup lives in
[`OpenMLE-Evo/third_party/aira-evo/src/dojo/tasks/mlebench/README.md`](OpenMLE-Evo/third_party/aira-evo/src/dojo/tasks/mlebench/README.md)
(clone `mle-bench`, `git lfs pull`, patch `data.py` line 29, then
`prepare.py -s all --data-dir=...`).

**Stated prerequisites** (`OpenMLE-Evo/docs/usage.md`) — this runtime orchestrates
search only; it does *not* serve models, prepare data, or ship sandbox images. You
must already have: Python 3.11/3.12, an OpenAI-compatible model service, a CPU/GPU
sandbox speaking the `/api/v1/jobs` protocol, the evaluation parquet + prepared
task data + leaderboard metadata, and the sandbox API keys.

---

## Tier 3b — The OpenMLE Sandbox cluster

If you deploy the distributed sandbox rather than running candidates locally,
[`OpenMLE-Gym/openmle-sandbox/README.md`](OpenMLE-Gym/openmle-sandbox/README.md)
**states** these requirements, validated on Ubuntu 22.04:

| Requirement | Detail |
| --- | --- |
| OS | Ubuntu 22.04, **linux/amd64** — the four published images are amd64 only, so **ARM VMs (Graviton, Ampere) will not work** |
| Docker | Docker Engine + Compose v2 plugin |
| GPU workers | NVIDIA driver, NVIDIA Container Toolkit, Docker GPU runtime; `nvidia-smi` working on the host |
| **cgroup v1** | Required by the 1.0.0 worker image (AIO Sandbox v0.9.2). Verify with `stat -fc %T /sys/fs/cgroup` → should report `tmpfs`, not `cgroup2fs`. Switching means editing GRUB (`systemd.unified_cgroup_hierarchy=0`) and **rebooting the host** |
| Local scratch | **≥100 GiB free** per worker, matching `LOCAL_SCRATCH_MIN_FREE_BYTES` |
| Shared storage | One POSIX/NFS filesystem mounted at the **same host path** on the controller and every worker |
| Client | Python 3.10+ |
| Time | Synchronized clocks across nodes |
| Ports | Controller `6580`; Router `6591`; worker range e.g. `10150-10157` |

Minimum useful topology (**stated**): one controller host (PostgreSQL + Redis +
FastAPI + dispatcher + Nginx) and one or more worker hosts. Start with **one
sandbox container per physical GPU** before trying oversubscription.

> The cgroup v1 requirement is the one that bites. It's a host-wide boot change
> requiring a reboot, and it conflicts with Kubernetes and most modern container
> setups. Use a dedicated worker host.

---

## Tier 4 — Training (SFT and RL)

**Stated** — `OpenMLE-ERL/SFT/docs/usage.md`:

> BF16 full-parameter training on **8 NVIDIA H200 GPUs**, global batch size 128,
> `3e-5` peak LR with cosine decay, 0.1 warmup fraction, three epochs, and a
> 32,768-token context limit.

Prerequisites listed there: Linux + Python 3.11+, uv, a CUDA environment, the
SLIME training image, a compatible sandbox evaluation service, **eight H200-class
GPUs**, and a converted Megatron `torch_dist` checkpoint.

**Stated** — `OpenMLE-ERL/RL/docs/usage.md`. Only the single-node async profile has
real runtime validation, on **8 H200 GPUs**:

| Profile | Model | GPUs |
| --- | --- | --- |
| Single-node sync | Qwen3-30B-A3B-Thinking-2507 | 1 × 8, colocated |
| Single-node async ✅ *validated* | Qwen3-30B-A3B-Thinking-2507 | 4 training + 4 rollout |
| Multi-node sync | Qwen3.6-35B-A3B | 2 × 8, colocated |
| Multi-node async | Qwen3.6-35B-A3B | 8 training + 8 rollout |

The other three profiles come from the same launch chain but "should be treated as
unvalidated until they complete equivalent real runs."

RL also needs a SLIME image supplying PyTorch/CUDA, Ray, SGLang, Megatron-LM, and
Transformer Engine, plus a working sandbox for reward scoring.

This tier is a multi-node cluster commitment, not a VM you rent for an afternoon.

---

## Recommendation

**Start at tier 1.** Provision one CPU VM — 8 vCPU / 32 GB / 100 GB, Ubuntu 22.04
— point `--model-base-url` at a hosted model, and run the NatureBench smoke. It
exercises the entire search loop end to end for the price of a few thousand API
tokens and costs nothing in GPU time. Only move to tier 2+ once you know which
experiment you actually want to reproduce.

Do not size a GPU box before you have a tier-1 run working. Nothing above tier 1
tells you anything a tier-1 run won't, until you're reproducing benchmark numbers.

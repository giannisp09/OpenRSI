# Setup

How to get this repo running on a fresh machine (VM or laptop). Every command
below was verified end to end before being written down.

Dependencies are managed with [**uv**](https://docs.astral.sh/uv/). The repo root
is a **uv workspace** whose members are `OpenMLE-Evo` and its vendored
`third_party/aira-evo`, plus the direct dependencies of `ctf_gym`. One lockfile
(`uv.lock`) pins all 261 packages, so every machine gets the same environment.

## TL;DR

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
gh repo clone giannisp09/OpenRSI && cd OpenRSI
uv sync                                            # creates .venv, ~2 min
make test                                          # Ran 2 tests ... OK
```

`uv sync` **downloads a matching Python for you**, so nothing else needs to be
installed first — no `deadsnakes`, no `python3.12-venv`, no `pip install`.

**Which VM should you rent?** See [`HARDWARE.md`](HARDWARE.md). Short version: the
`ctf_gym` suite and a complete single-task NatureBench search both run on a
**CPU-only** VM (8 vCPU / 32 GB / 100 GB, Ubuntu 22.04) when the model comes from
a hosted API. You only need GPUs to self-host Frontis-MA1 or to reproduce
MLE-Bench.

## 0. Prerequisites

| Requirement | Why |
| --- | --- |
| **uv** | Creates the environment and the interpreter. Install: `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then restart the shell (or `source ~/.local/bin/env`). |
| Git | Cloning. |
| Docker | Only to build/run the `ctf_gym` benchmark targets (`ctf_gym/benchmarks/*/Dockerfile`). Not needed for the unit tests. |

You do **not** need a system Python. `.python-version` pins **3.12**, and uv
fetches that interpreter if the machine does not have one.

> **Python 3.13 will not work.** `OpenMLE-Evo` declares
> `requires-python = ">=3.11,<3.13"`. uv enforces this automatically; a manual
> `python3 -m venv` on a 3.13 host does not, and fails later at install time.

## 1. Clone the private repo

**Option A — GitHub CLI (recommended, no token to manage):**

```bash
gh auth login          # GitHub.com -> HTTPS -> authenticate in browser
gh repo clone giannisp09/OpenRSI
```

**Option B — Personal Access Token:**

Create a classic PAT at *Settings > Developer settings > Personal access tokens >
Tokens (classic)* with the **`repo`** scope, then:

```bash
git clone https://github.com/giannisp09/OpenRSI.git
# Username: giannisp09
# Password: <paste the PAT, not your account password>
```

To avoid re-entering it: `git config --global credential.helper store`.

> The repo has some large tracked blobs (`OpenMLE-Gym/builder_core/info.csv`
> 39 MB, `docs/assets/videos/*.mp4` 19 MB). `git clone --filter=blob:none` gives
> a much faster partial clone if you only need the code.

## 2. Create the environment

```bash
cd OpenRSI
uv sync                # runtime deps
uv sync --all-groups   # ...plus the dev group (pytest)
```

This creates `./.venv` at the repo root and installs `openmle-tts-eval` and
`aira-dojo` in editable mode from the local checkout. Do not create a venv by
hand and do not `pip install` into it — `uv sync` owns `.venv` and will reconcile
it against `uv.lock` on every run.

Run commands with `uv run`, which syncs first and needs no activation:

```bash
uv run python -m unittest ctf_gym.tests.test_ctf_gym -v
```

Or activate the venv the usual way if you prefer: `source .venv/bin/activate`.

### Common tasks

| Command | Does |
| --- | --- |
| `make setup` | `uv sync --all-groups` |
| `make test` | Runs the `ctf_gym` suite with `LOGGING_DIR` set |
| `make lock` | Re-resolves `uv.lock` after editing a `pyproject.toml` |
| `make upgrade` | Re-resolves, allowing newer versions |
| `make export` | Regenerates the pip-compatible `requirements.txt` |
| `make clean` | Deletes `.venv` and caches |

### Adding a dependency

Edit `pyproject.toml` (or a member's), then:

```bash
uv add somepackage     # or: edit pyproject.toml && make lock
make export            # keep the pip fallback in sync
```

Commit both `pyproject.toml` and `uv.lock`.

### No-uv fallback

`requirements.txt` at the repo root is **generated from `uv.lock`** by
`make export` and is fully pinned, so plain pip reproduces the same environment:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

Do not hand-edit it; edit `pyproject.toml` and re-run `make export`.

## 3. Configure environment variables

`dojo` reads its configuration from environment variables and fails at **import
time** if they are missing.

```bash
cp OpenMLE-Evo/.env.example OpenMLE-Evo/.env
# then edit the copy
```

Variables `dojo` can demand (from `dojo/utils/environment.py`):

| Variable | Needed for |
| --- | --- |
| `LOGGING_DIR` | **Always** — required to import the task modules at all. |
| `MLE_BENCH_DATA_DIR` | MLE-Bench task data. |
| `SUPERIMAGE_DIR` | Container image staging. |
| `DEFAULT_SLURM_ACCOUNT` / `DEFAULT_SLURM_PARTITION` / `DEFAULT_SLURM_QOS` | Slurm-backed runs only. |

Model/API credentials also live in that file. **Never commit it** — it is
gitignored; only the `.example` template is tracked.

uv can load it for a single command:

```bash
uv run --env-file OpenMLE-Evo/.env python -m unittest ctf_gym.tests.test_ctf_gym
```

## 4. Verify the install

```bash
# from the repo ROOT (ctf_gym uses PEP 420 namespace packages, so cwd matters)
make test
```

Expected: `Ran 2 tests ... OK` (~13s; the suite starts a local uvicorn server on
port 18556). `make test` points `LOGGING_DIR` at `./.logs` for you.

If that passes, the environment is correct and you are ready to run experiments.

## 5. Run your first experiment

### Option A — CTF gym (no GPU, no model, no data)

Already covered by `make test`. To exercise the benchmark targets as containers:

```bash
docker build -t deepred-target ctf_gym/benchmarks/deepred/target_service
docker build -t exploitbench-324747822 ctf_gym/benchmarks/exploitbench/324747822
```

See [`CTF-GYM-SPEC.md`](CTF-GYM-SPEC.md) for the task and verifier contract.

### Option B — One NatureBench task (no GPU, hosted model)

The smallest complete OpenMLE-Evo search loop. Full instructions are in
[`OpenMLE-Evo/benchmarks/naturebench_local_quick/README.md`](OpenMLE-Evo/benchmarks/naturebench_local_quick/README.md);
the short path:

```bash
# 1. NatureBench data, cloned NEXT TO the OpenRSI checkout
cd .. && git clone https://github.com/FrontisAI/NatureBench.git && cd OpenRSI

# 2. Conda runtime for the generated candidate code (separate from .venv)
cd OpenMLE-Evo
conda env create -f environments/naturebench-local.yml
cp .env.example .env

# 3. Run one candidate against a hosted OpenAI-compatible endpoint
export PRIMARY_KEY='your-api-key'
../.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ../../NatureBench \
  --conda-env naturebench-local \
  --model-base-url https://model.example/v1 \
  --model-id served-model-name \
  --smoke
```

Note `../.venv/bin/python` — the workspace venv lives at the repo root now, not
inside `OpenMLE-Evo/`. Drop `--smoke` for the full single-task search (4 h
effective budget, 6 h wall-clock, up to 160 nodes); `--smoke` caps it at one
candidate and a 30-minute budget.

> ⚠️ **Run this on a disposable VM.** Conda isolates Python dependencies, not
> files, networking, or host permissions — model-generated code executes with
> your user's full access. Do not run it on a machine holding credentials or
> anything you care about.

### Option C — MLE-Bench

Needs prepared Kaggle data, a sandbox speaking the `/api/v1/jobs` protocol, and a
GPU sandbox worker. Start from `OpenMLE-Evo/docs/usage.md`; the entry point is
`./scripts/run_standard.sh` (or `run_multi_gpu.sh` for the async profile), run
with the workspace venv active:

```bash
cd OpenMLE-Evo && source ../.venv/bin/activate && ./scripts/run_standard.sh
```

See [`HARDWARE.md`](HARDWARE.md) §3 before committing to this one.

## 6. The other sub-projects

`OpenMLE-Gym` and `OpenMLE-ERL/SFT` are **not** workspace members. They are
independent projects with their own `uv.lock` files — and `OpenMLE-ERL/SFT` uses
a different package index and has mutually conflicting extras, so folding it in
would make the root resolution unsolvable. Set them up on their own:

```bash
cd OpenMLE-Gym     && uv sync    # or: uv sync --extra all
cd OpenMLE-ERL/SFT && uv sync    # see its README for the extras to pick
```

`OpenMLE-ERL/RL` installs inside a SLIME runtime image; see
`OpenMLE-ERL/RL/README.md`.

## 7. Git remotes on the VM

A clone of the private repo gets `origin` pointing at it, which is what you
want. To also track the public upstream:

```bash
git remote add upstream https://github.com/FrontisAI/OpenRSI.git
git remote set-url --push upstream DISABLED   # guard against pushing private work to the public repo
```

## Gotchas

- **Namespace packages:** `ctf_gym` has no `__init__.py` files. Imports only
  resolve when the repo root is on `sys.path` — run from the repo root (which
  `uv run` and `make` both do), or set `PYTHONPATH=/path/to/OpenRSI`.
- **Don't copy `.venv` from your laptop.** macOS venvs contain Darwin binaries
  (`.dylib`, `*-darwin.so`) that will not load on Linux. They are gitignored for
  this reason; always `uv sync` on the VM.
- **`OpenMLE-Evo/.runtime/`** is a second, separately-managed virtualenv used by
  the runner. It is gitignored and regenerated on demand — do not copy it either.
- **`OpenMLE-Evo/requirements.txt`** (`-e .`, `-e ./third_party/aira-evo`) is
  kept for that sub-project's own standalone pip flow. The workspace does not use
  it; `uv sync` at the root installs both members.

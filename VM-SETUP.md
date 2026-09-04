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

For the OpenMLE-Evo side, see `OpenMLE-Evo/README.md` — its entry point is
`./scripts/run_standard.sh`. Run it with the workspace venv active, e.g.
`cd OpenMLE-Evo && source ../.venv/bin/activate && ./scripts/run_standard.sh`.

## 5. The other sub-projects

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

## 6. Git remotes on the VM

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

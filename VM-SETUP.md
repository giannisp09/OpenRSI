# VM Setup

How to get this repo running on a fresh Linux VM. Every command below was
verified against the working macOS environment before being written down.

## 0. Prerequisites

| Requirement | Why |
| --- | --- |
| **Python 3.11 or 3.12** | `OpenMLE-Evo` declares `requires-python = ">=3.11,<3.13"`. **Python 3.13 will not work** — the `aira_dojo` / `openmle-tts-eval` editable installs are rejected. |
| Git | Cloning. |
| Docker | Only needed to build/run the `ctf_gym` benchmark targets (`ctf_gym/benchmarks/*/Dockerfile`). Not needed for the unit tests. |

```bash
python3.12 --version   # must print 3.11.x or 3.12.x
```

If the VM's default `python3` is 3.13, install 3.12 explicitly, e.g. on Ubuntu:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv
```

## 1. Clone the private repo

The repo is **private**, so the clone must authenticate.

**Option A — GitHub CLI (recommended, no token to manage):**

```bash
gh auth login          # choose GitHub.com -> HTTPS -> authenticate in browser
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

## 2. Create the environment

There is **no root-level `requirements.txt`** — this is a monorepo of
sub-projects. Create **one** virtualenv at the repo root and install both
dependency sets into it, because `ctf_gym` imports `dojo` from `OpenMLE-Evo`.

```bash
cd OpenRSI
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r OpenMLE-Evo/requirements.txt   # installs openmle-tts-eval + aira_dojo (editable)
pip install -r ctf_gym/requirements.txt       # fastapi, uvicorn, requests
```

> **Why two files:** `OpenMLE-Evo/requirements.txt` provides the `dojo` package
> that `ctf_gym/src/base_task.py` imports; `ctf_gym/requirements.txt` provides
> the web deps that `ctf_gym/src/verifier/router.py` imports. Installing only
> one of them leaves the `ctf_gym` suite unimportable.

## 3. Configure environment variables

`dojo` reads its configuration from environment variables and fails at **import
time** if they are missing.

```bash
cp OpenMLE-Evo/.env.example OpenMLE-Evo/.env
# then edit OpenMLE-Evo/.env
```

Variables `dojo` can demand (from `dojo/utils/environment.py`):

| Variable | Needed for |
| --- | --- |
| `LOGGING_DIR` | **Always** — required to import the task modules at all. |
| `MLE_BENCH_DATA_DIR` | MLE-Bench task data. |
| `SUPERIMAGE_DIR` | Container image staging. |
| `DEFAULT_SLURM_ACCOUNT` / `DEFAULT_SLURM_PARTITION` / `DEFAULT_SLURM_QOS` | Slurm-backed runs only. |

Model/API credentials also live in that file. **Never commit it** — `.env` is
gitignored; only `.env.example` is tracked.

## 4. Verify the install

```bash
# from the repo ROOT (ctf_gym uses PEP 420 namespace packages, so cwd matters)
export LOGGING_DIR=/tmp/openrsi-logs && mkdir -p "$LOGGING_DIR"

python -c "import ctf_gym.src.verifier.router, ctf_gym.src.tasks.deepred_task; print('imports OK')"
python -m unittest ctf_gym.tests.test_ctf_gym -v
```

Expected: `Ran 2 tests ... OK` (takes ~13s; the suite starts a local uvicorn
server on port 18556).

For the OpenMLE-Evo side, see `OpenMLE-Evo/README.md` — its entry point is
`./scripts/run_standard.sh`.

## 5. Git remotes on the VM

A clone of the private repo gets `origin` pointing at it, which is what you
want. To also track the public upstream:

```bash
git remote add upstream https://github.com/FrontisAI/OpenRSI.git
git remote set-url --push upstream DISABLED   # guard against pushing private work to the public repo
```

## Gotchas

- **Namespace packages:** `ctf_gym` has no `__init__.py` files. Imports only
  resolve when the repo root is on `sys.path` — run from the repo root, or set
  `PYTHONPATH=/path/to/OpenRSI`.
- **Don't copy `.venv` from your laptop.** The macOS venvs contain Darwin
  binaries (`.dylib`, `*-darwin.so`) that will not load on Linux. They are
  gitignored for this reason; always rebuild on the VM.
- **`OpenMLE-Evo/.runtime/`** is a second, separately-managed virtualenv used by
  the runner. It is gitignored and regenerated on demand — do not copy it either.
- **Large tracked files:** `OpenMLE-Gym/builder_core/info.csv` (39 MB) and
  `docs/assets/videos/*.mp4` (19 MB) make the initial clone slow. `git clone
  --filter=blob:none` gives a faster partial clone if you only need code.

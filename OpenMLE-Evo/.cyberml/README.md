# CyberML gym — cybersecurity ML tasks for OpenMLE-Evo

Cybersecurity supervised-ML tasks wired to the OpenMLE-Evo search loop (Draft /
Improve / Debug / Crossover + island search + experience memory). Each is an
ordinary ML task with an ungameable verifier, so they reuse the existing
NatureBench runner unchanged — **no new Task subclass, no changes to the search,
and `ctf_gym/` is untouched** (offensive/CTF is the next step).

**Six tasks** ship today, spanning distinct cyber domains and feature spaces but
one interface (`predictions.csv` with a `label_pred` column, metric
`Detection F1-Score`) — a curriculum for testing experience transfer and for
measuring self-improvement across a *plethora* of cyber-ML tasks:

| task id | domain | feature space | instances | weak-LR baseline F1 |
| --- | --- | --- | --- | --- |
| `nsl-kdd-nids`      | network intrusion (NSL-KDD) | 41 flow features | dos, probe, r2l, u2r | 0.89 / 0.87 / 0.03 / 0.57 |
| `unsw-nb15-nids`    | network intrusion (UNSW-NB15) | 42 flow features + 3 categoricals | main | 0.91 |
| `phishing-url`      | phishing detection | 111 URL lexical/host features | main | 0.91 |
| `dga-domains-dns`   | DGA / malware C2 domains | raw domain **string** (engineer features) | main | 0.67 |
| `clamp-pe-malware`  | static PE-header malware | 68 PE-header features | main | 0.965 |
| `tuandromd-android` | Android malware | 241 permission/API 0/1 flags | main | 0.995 |

The spread is deliberate: `dga-domains-dns` (0.67) and `nsl-kdd-nids`'s r2l/u2r
carry the most headroom; `tuandromd-android` (0.995) and `clamp-pe-malware`
(0.965) are near-saturated and mostly test **preservation**; `dga-domains-dns` is
the only task with a raw string input, so it tests **feature engineering** rather
than model choice. Running them in sequence with experience memory on tests
whether operator experience transfers across cybersecurity domains — the
curriculum / RSI angle. Use `tools/measure_improvement.py` (below) to report
per-task and overall improvement after a run.

## The rest of this doc uses `nsl-kdd-nids` as the running example.

The task is deliberately split into **four attack-family instances** — `dos`,
`probe`, `r2l`, `u2r` — spanning the difficulty range so the search reproduces the
per-component / preservation dynamic from `PORTING-OPENMLE.md` §4:

| instance | weak-LR baseline F1 | role |
| --- | --- | --- |
| `dos`   | ~0.89 | easy, saturates early |
| `probe` | ~0.87 | easy, saturates early |
| `r2l`   | ~0.03 | hard — most of the headroom |
| `u2r`   | ~0.57 | hard, only ~67 test positives |

`aggregate_improvement` is the mean over the four, so a gain on `r2l`/`u2r` that
regresses `dos`/`probe` nets out negative — the README states the preservation
constraint explicitly.

## Files

```
.cyberml/
  tools/prepare_nsl_kdd.py          # download NSL-KDD, build the 4 instances
  tools/prepare_unsw_nb15_nids.py   # download UNSW-NB15, build the task
  tools/prepare_phishing_url.py     # download phishing-URL features, build the task
  tools/prepare_dga_domains_dns.py  # download DGA domains, build the task
  tools/prepare_clamp_malware.py    # download ClaMP, build the malware task
  tools/prepare_tuandromd_android.py # download TUANDROMD, build the task
  tools/selfcheck.py                # offline verifier proof (no model / no NatureBench)
  tools/test_local_eval_service.py  # offline test of the local eval service
  tools/measure_improvement.py      # per-task + overall improvement across runs
  local_naturebench/                # self-contained eval service (no external repo)
    eval_service.py                 # /health /register /start_timer /evaluate
    run_naturebench.py              # stub to satisfy the runner's repo check
  data/tasks/<task-id>/
    metadata.json            # metric = "Detection F1-Score", per-instance anchors
    problem/README.md        # task + strict output contract + starter run.py
    problem/data_description.md
    problem/data/<inst>/{train.csv,x_test.csv}          # visible
    evaluation/evaluator.py                             # sklearn binary F1
    evaluation/ground_truth/<inst>/y_ref.csv            # hidden labels
    licenses/ATTRIBUTION.txt
benchmarks/cyberml_nsl_kdd/tasks.txt        # single-task set
benchmarks/cyberml_curriculum/tasks.txt     # 2-task transfer curriculum
benchmarks/cyberml_all/tasks.txt            # all six tasks (full sweep)
```

## Step 0 — build data + verify offline (no GPU, no model, no NatureBench)

From the repo root:

```bash
uv sync
uv run python OpenMLE-Evo/.cyberml/tools/prepare_nsl_kdd.py          # NSL-KDD intrusion
uv run python OpenMLE-Evo/.cyberml/tools/prepare_unsw_nb15_nids.py   # UNSW-NB15 intrusion
uv run python OpenMLE-Evo/.cyberml/tools/prepare_phishing_url.py     # phishing-URL
uv run python OpenMLE-Evo/.cyberml/tools/prepare_dga_domains_dns.py  # DGA domains
uv run python OpenMLE-Evo/.cyberml/tools/prepare_clamp_malware.py    # ClaMP PE malware
uv run python OpenMLE-Evo/.cyberml/tools/prepare_tuandromd_android.py # TUANDROMD Android
uv run python OpenMLE-Evo/.cyberml/tools/selfcheck.py               # verifier -> SELF-CHECK: PASS
uv run python OpenMLE-Evo/.cyberml/tools/test_local_eval_service.py  # eval svc -> EVAL-SERVICE TEST: PASS
```

`selfcheck.py` runs the starter as a "good" submission, plus an all-zero and a
malformed submission, and asserts the verifier scores/rejects each correctly
(`PORTING-OPENMLE.md` §6 "verifier first, alone"). `test_local_eval_service.py`
boots the local eval service and checks the full register/evaluate contract. Both
validate every task under `data/tasks/` automatically.

## Step 1 — run the real evo search

Needs an OpenAI-compatible model endpoint, plus the eval service that turns
evaluator metrics into `aggregate_improvement`. Two ways to get the eval service:

**Option 1 (recommended) — local eval service, no external repo.** Point
`--naturebench-repo` at the bundled `local_naturebench/` (validated by
`test_local_eval_service.py` above).

**Runtime for candidate code:** use the existing uv `.venv` directly with
`--local-python .venv/bin/python`. It already has numpy/scipy/pandas/scikit-learn
(and fastapi/uvicorn for the eval service), so **you do not need to build the
`naturebench-local` conda env** on this machine. (`--conda-env` is the alternative
for a Linux box where you want an isolated candidate runtime.)

**API key:** put `PRIMARY_KEY=...` in `OpenMLE-Evo/.env` — the runner now
auto-loads `.env` (cwd first, then repo root) and prints which keys it loaded
(values hidden). An already-exported `PRIMARY_KEY` still wins. Use `EMPTY` for a
keyless local model.

```bash
cd OpenMLE-Evo
# PRIMARY_KEY lives in .env (auto-loaded); no export, no conda env needed.
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo .cyberml/local_naturebench \
  --local-python .venv/bin/python \
  --data-dir .cyberml/data --skip-download \
  --task nsl-kdd-nids \
  --model-base-url https://openrouter.ai/api/v1 \
  --model-id anthropic/claude-sonnet-5 \
  --smoke
```

The **base-url must match the provider your `PRIMARY_KEY` belongs to** — an
OpenRouter key (`sk-or-...`) → `https://openrouter.ai/api/v1` with a full slug
like `anthropic/claude-sonnet-5`; an OpenAI key → `https://api.openai.com/v1`;
a local model → `http://127.0.0.1:<port>/v1` with `PRIMARY_KEY=EMPTY`.

**Option 2 — external NatureBench repo** (provides its own `eval_service.py`):

```bash
git clone https://github.com/FrontisAI/NatureBench.git   # sibling of OpenRSI
cd OpenMLE-Evo
conda env create -f environments/naturebench-local.yml    # candidate runtime
cp .env.example .env
```

### Model endpoint — pick one

The runner already speaks to any OpenAI-compatible `/v1` endpoint — **no code
change** is needed to swap an API for a self-hosted model.

**(A) Self-hosted Frontis-MA1-35B on a rented GPU — no API key.** Serve it with
vLLM (`scripts/serve_frontis_ma1.sh` on the GPU box) and tunnel in; the runner
auto-sets `PRIMARY_KEY=EMPTY` for the no-auth server:

```bash
# GPU box: ./scripts/serve_frontis_ma1.sh   (vLLM, --served-model-name frontis-ma1)
MODEL_FLAGS="--model-ssh-host <gpu-host> --model-ssh-port 8000 --model-id frontis-ma1"
```

The full GPU-sizing table (BF16 ~70 GB → H100-80 / FP8 ~35 GB → A100-40 / GGUF
Q4 ~20 GB → 24 GB card or this M4 Pro), the serve commands, both wiring paths
(SSH-tunnel and direct `--model-base-url`), reasoning-parser and concurrency
tuning, and the CC BY-NC 4.0 license note are in **[`LOCAL_MODEL.md`](LOCAL_MODEL.md)**.

**(B) Local GGUF on Apple MPS — no API key.** The GGUF Q4 35B (~18–20 GB) fits in
this M4 Pro's 48 GB unified memory (Ollama / llama.cpp exposing an OpenAI `/v1`):

```bash
export PRIMARY_KEY=EMPTY
MODEL_FLAGS="--model-base-url http://127.0.0.1:11434/v1 --model-id <served-name>"
```

**(C) Hosted API instead:**

```bash
export PRIMARY_KEY='your-api-key'   # or put it in .env (auto-loaded)
MODEL_FLAGS="--model-base-url https://model.example/v1 --model-id served-model-name"
```

### Smoke (one candidate — proves plumbing, crash→Debug, partial-score→Improve)

```bash
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo .cyberml/local_naturebench \
  --local-python .venv/bin/python \
  --data-dir .cyberml/data --skip-download \
  --task nsl-kdd-nids \
  $MODEL_FLAGS \
  --smoke
```

### Bounded climb (~15 nodes — confirm the score actually rises)

Drop `--smoke` and cap the budget (default is 160 nodes / 6 h) with a Hydra
override after `--`:

```bash
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo .cyberml/local_naturebench \
  --local-python .venv/bin/python \
  --data-dir .cyberml/data --skip-download \
  --task nsl-kdd-nids \
  $MODEL_FLAGS \
  -- search.runner.solver.step_limit=15
```

Then read the trajectory row by row (`PORTING-OPENMLE.md` §5). Each candidate's
`aggregate_improvement`, `per_instance_improvement`, and raw `Detection F1-Score`
are logged in the run's journal:

```
output/<run>/program_ep_0/nsl-kdd-nids/aira_evo/checkpoint/journal.jsonl
```

Per-step feedback (what the Improve/Debug operator saw) is under
`output/<run>/program_ep_0/nsl-kdd-nids/step_<n>/feedback.txt`, and node fitness
is printed live as `Registering Node … Metric: <aggregate_improvement>`.

Confirm best-so-far F1 climbs (mostly via `r2l`/`u2r`) **and** that `dos`/`probe`
stay solved — the preservation behavior (Arm C in `PORTING-OPENMLE.md` §4). In a
validated 15-node run, best `aggregate_improvement` climbed 0.074 → 0.288, with
r2l F1 0.034 → 0.35 and u2r 0.57 → 0.82; `probe` regressed slightly (0.87 → 0.85),
showing the preservation constraint is only partly satisfied at this budget.

## Security boundary

Model-generated candidate code runs in whatever interpreter you pass to
`--local-python` (here the uv `.venv`) with **your user's full local access** — a
venv/Conda isolates dependencies, not files/network/processes. The Docker/SCM
sandbox profiles are linux/amd64 only and won't run on Apple Silicon, so this
local path is the one that works here. Run it on a disposable VM, or accept the
risk knowingly on a trusted machine (the NSL-KDD candidates are plain
sklearn/torch training, but the boundary is real).

## Known risk — resolved

`_verify_task_packages` imposes no task-name allow-list, so these packages pass
the runner's checks. The one previously-unconfirmed piece was whether an external
`eval_service.py` would register an arbitrary `task_name`. That risk is now
removed by **Option 1** above: `local_naturebench/eval_service.py` is a
self-contained, data-driven eval service (any task package with a `metadata.json`
+ `evaluation/evaluator.py` works), verified end-to-end offline by
`tools/test_local_eval_service.py`. The external repo (Option 2) remains
available but is no longer required.

The in-repo `evaluate_naturebench.py` + `NatureBenchTask` search half is also
**validated end-to-end**: a 15-node run on `nsl-kdd-nids` (Claude Sonnet 5 via
OpenRouter) climbed best `aggregate_improvement` 0.074 → 0.288, with r2l F1
0.034 → 0.35 and u2r 0.57 → 0.82. The five other tasks share the identical package
structure and pass the offline checks, so they are ready to run the same way.

## Running the full sweep + curriculum (experience transfer)

`benchmarks/cyberml_all/tasks.txt` lists all six tasks;
`benchmarks/cyberml_curriculum/tasks.txt` is the 2-task transfer demo. Point
`--task-set` at either (or pass `--task` repeatedly). Experience memory is on in
the default `airaevo_naturebench` profile, so operator experience accrued on one
task is available on the next:

```bash
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo .cyberml/local_naturebench \
  --local-python .venv/bin/python \
  --data-dir .cyberml/data --skip-download \
  --task-set benchmarks/cyberml_all/tasks.txt \
  $MODEL_FLAGS \
  -- search.runner.solver.step_limit=15
```

## Measuring improvement — per task and overall

After one or more runs, aggregate the results:

```bash
uv run python OpenMLE-Evo/.cyberml/tools/measure_improvement.py \
  --output-dir OpenMLE-Evo/output --json summary.json --csv summary.csv
```

It scans every run's journal
(`output/<run>/program_ep_*/<task-id>/aira_evo/checkpoint/journal.jsonl`), and for
each cyber task reports the **best `aggregate_improvement`** reached and the
winning candidate's per-instance raw F1 vs. the baseline anchor; then it prints
the **overall** mean best `aggregate_improvement` across tasks (the single "how
much did AI improve AI across the gym" number) and the spread. A task run more
than once keeps its best; `--all-tasks` includes non-cyber journals too. Because
`aggregate_improvement` is normalized per task (0 = that task's weak baseline,
1 = its SOTA anchor), the overall mean is comparable across the different metrics
and difficulty levels.

### Paper-ready results (tables, figures, manifest, report)

For a write-up, `make_paper_report.py` turns the same journals into the full
evidence pack:

```bash
uv run python OpenMLE-Evo/.cyberml/tools/make_paper_report.py \
  --model-label "Frontis-MA1-35B (local)"
# writes OpenMLE-Evo/.cyberml/paper/{tables,plots,manifest.json,REPORT.md}
```

It emits, under `.cyberml/paper/`:

- **`tables/`** — `main_results`, `per_instance`, `operator_effectiveness`,
  `compute_budget` as **CSV + LaTeX booktabs** (baseline/best/SOTA, normalized
  improvement, best-node step & operator, buggy %, wall-clock, tokens incl.
  reasoning tokens, and real per-run **cost** read from the journal).
- **`plots/`** — publication PDF+PNG: per-task **search trajectory** (nodes
  colored by operator, best-so-far envelope, baseline/SOTA reference lines),
  a small-multiples `trajectory_all`, an `aggregate_bar` across tasks (with
  cross-seed error bars when a task was run more than once), `operator_effectiveness`,
  and per-instance `family_<task>` bars.
- **`manifest.json`** — git commit, host, package versions, and the exact run
  dirs consumed (reproducibility).
- **`REPORT.md`** — a methods+results write-up that embeds the tables, references
  the figures, includes a representative model reasoning trace, and prints the
  reproduce command.

`--price-input/--price-output` (USD per 1M tokens) recompute cost from token
counts (leave unset to use the recorded cost; a local model records \$0). Run
each task with **multiple seeds** (repeat the run) for mean±std — the
`mean_across_runs`/`std_across_runs` columns and the bar-chart error bars populate
automatically. For larger, more rigorous runs raise the budget
(`search.runner.solver.step_limit=40` or higher) — see `LOCAL_MODEL.md` for a
self-hosted-model sweep.

## Adding still more cyber-ML tasks

Each new task (EMBER/BODMAS malware, CIC-IDS flows, TON-IoT, TLS-fingerprint, …)
is the same recipe: author a package under `data/tasks/<id>/` (`metadata.json` +
`problem/README.md` with a starter + train/test data + `evaluation/evaluator.py`
with hidden labels), add `<id>` to a task-set file, and run. The six prepare
scripts in `tools/` are templates — copy the closest one (single-instance numeric
= `prepare_phishing_url.py`; with categoricals = `prepare_unsw_nb15_nids.py`;
raw-string features = `prepare_dga_domains_dns.py`; multi-instance =
`prepare_nsl_kdd.py`), swap the data source, schema, and instances. The search,
operators, and experience memory are reused unchanged, so testing self-improvement
across a *plethora* of cyber-ML tasks is additive, one package at a time.
Experience carrying across tasks (and later into `ctf_gym`) is the RSI angle.

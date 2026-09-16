# Local NatureBench-compatible eval service

This directory lets the OpenMLE-Evo NatureBench runner drive a search **without
cloning the external NatureBench repo**. The runner accepts a `--naturebench-repo`
directory that contains `run_naturebench.py` + `eval_service.py`; this is a
minimal, self-contained implementation of both, for locally-authored task
packages (the `.cyberml/` cyber-ML tasks).

- `eval_service.py` — a FastAPI service implementing the exact HTTP contract the
  in-repo `NatureBenchTask` client speaks (`/health`, `/register`, `/start_timer`,
  `/evaluate`). On `/evaluate` it runs the task package's own
  `evaluation/evaluator.py` and converts the raw metrics in `score.json` into
  per-instance *improvements* against the `baseline_score`/`sota_score` anchors in
  the package's `metadata.json`, returning `aggregate_improvement` (the mean),
  `best_aggregate_improvement`, `per_instance_improvement`, and `raw_scores`.
- `run_naturebench.py` — a stub that satisfies the runner's repo check. It is only
  invoked when you omit `--skip-download`; in that case it just verifies the
  requested packages are already materialized (it never downloads).

## Improvement formula

```
improvement_i = (score_i - baseline_i) / (sota_i - baseline_i)
```

0.0 = baseline, 1.0 = SOTA, > 1.0 = above SOTA; if `sota <= baseline` it falls
back to raw gain over baseline. This is a documented, gradient-preserving local
approximation of NatureBench's `aggregate_improvement` — the exact upstream
formula is not needed for the search to climb, and the anchors are per-task in
each package's `metadata.json`, so tune them there.

## Use it

Point the runner's `--naturebench-repo` at this directory and pass
`--skip-download`:

```bash
cd OpenMLE-Evo
# PRIMARY_KEY is auto-loaded from .env; candidate code runs in the uv .venv.
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo .cyberml/local_naturebench \
  --local-python .venv/bin/python \
  --data-dir .cyberml/data --skip-download \
  --task-set benchmarks/cyberml_curriculum/tasks.txt \
  --model-base-url <your /v1 endpoint> --model-id <model> \
  --smoke
```

Note: `--data-dir` is the directory that **contains** `tasks/` (so `.cyberml/data`,
not `.cyberml/data/tasks`) — the task builder appends `tasks/<id>` itself.

The runner boots this service on `--eval-host:--eval-port` (default
`127.0.0.1:8321`), reads the control token this service writes to
`eval_logs/eval_control_token`, and drives the search through the in-repo
`evaluate_naturebench.py` + `NatureBenchTask`, which talk to this service.

## Verify offline (no model, no external repo)

```bash
uv run python OpenMLE-Evo/.cyberml/tools/test_local_eval_service.py
# boots the service, registers each task, evaluates good + all-zero submissions,
# asserts aggregate_improvement is sane -> EVAL-SERVICE TEST: PASS
```

## Boundaries

- Like NatureBench's own local mode this scores model-written output on the host;
  it does **not** sandbox. Use only with the local execution profile on a
  trusted/disposable machine.
- Control-token gating is written but not enforced (local single-user). Do not
  expose this service off loopback.
- This covers the eval-service half of the loop. The task-description builder
  (`build_tasks.py`) and search (`evaluate_naturebench.py`) are the in-repo
  aira-evo components and are used unchanged.

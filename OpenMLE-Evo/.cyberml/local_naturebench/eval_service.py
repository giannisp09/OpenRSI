#!/usr/bin/env python3
"""Local, self-contained NatureBench-compatible eval service.

Purpose: let the OpenMLE-Evo NatureBench runner drive a search **without cloning
the external NatureBench repo**. The runner requires a ``--naturebench-repo``
directory containing ``run_naturebench.py`` + ``eval_service.py`` and boots the
latter as ``python eval_service.py --host H --port P``; this file is that
service, for locally-authored task packages (e.g. the .cyberml/ cyber-ML tasks).

It implements exactly the HTTP contract that the in-repo NatureBenchTask client
speaks (third_party/aira-evo/examples/nature_bench/base_task.py):

  GET  /health       -> {"status": "ok"}
  POST /register     {task_name, data_dir, timeout, out_dir, batch_name, eval_token}
  POST /start_timer  {task_name, batch_name}
  POST /evaluate     {task_name, batch_name, output_dir, eval_token}
                     -> {aggregate_improvement, best_aggregate_improvement,
                         per_instance_improvement, raw_scores}

Scoring: on /evaluate the service runs the *package's own*
``evaluation/evaluator.py`` (which reads OUTPUT_DIR and compares to hidden ground
truth), then converts the raw per-instance metrics in ``score.json`` into
per-instance *improvements* against the ``baseline_score`` / ``sota_score``
anchors in the package's ``metadata.json``. ``aggregate_improvement`` is the mean
over instances.

Improvement normalization (documented, and intentionally simple):

    improvement_i = (score_i - baseline_i) / (sota_i - baseline_i)

so 0.0 == baseline, 1.0 == SOTA, > 1.0 == above SOTA. This is a faithful,
gradient-preserving local approximation of NatureBench's aggregate_improvement;
the exact upstream formula is not needed for the search to climb, and the anchors
live in each task's metadata.json (tune them there). See .cyberml/README.md.

NOTE ON ISOLATION: like NatureBench's own local mode, this scores model-written
output on the host. It does not sandbox. Run it only alongside the local
execution profile on a trusted/disposable machine.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Local NatureBench-compatible eval service")

_LOCK = threading.Lock()
# task_name -> {data_dir, out_dir, timeout, started_at, best_aggregate}
_REGISTRY: dict[str, dict[str, Any]] = {}
SERVICE_DIR = Path(__file__).resolve().parent
CONTROL_TOKEN = ""  # set at startup; accepted-but-not-required (local single-user)


# --------------------------------------------------------------------------- #
# Payload models (lenient: extra keys ignored, unknown-but-present tolerated)  #
# --------------------------------------------------------------------------- #
class RegisterBody(BaseModel):
    task_name: str
    data_dir: str
    out_dir: str | None = None
    timeout: float | None = None
    batch_name: str | None = None
    eval_token: str | None = None
    force: bool | None = None

    class Config:
        extra = "allow"


class TimerBody(BaseModel):
    task_name: str
    batch_name: str | None = None

    class Config:
        extra = "allow"


class EvaluateBody(BaseModel):
    task_name: str
    output_dir: str | None = None
    batch_name: str | None = None
    eval_token: str | None = None

    class Config:
        extra = "allow"


# --------------------------------------------------------------------------- #
# Scoring helpers                                                             #
# --------------------------------------------------------------------------- #
def _primary_anchors(metadata: dict) -> dict[str, dict[str, float]]:
    """{instance: {"baseline": float, "sota": float}} for the primary metric."""
    anchors: dict[str, dict[str, float]] = {}
    for entry in metadata.get("performance_entries", []):
        inst = entry.get("dataset_name")
        primary = None
        for metric in entry.get("metrics", []):
            if metric.get("is_primary"):
                primary = metric
                break
        if inst is None or primary is None:
            continue

        def _num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        base = _num((primary.get("baseline_score") or {}).get("value"))
        sotas = primary.get("sota_score") or []
        sota = _num(sotas[0].get("value")) if sotas else None
        anchors[inst] = {"baseline": base, "sota": sota}
    return anchors


def _improvement(score: float | None, base: float | None,
                 sota: float | None) -> float | None:
    if score is None:
        return None
    if base is None:
        base = 0.0
    if sota is None or sota <= base:
        # No usable ceiling: fall back to raw gain over baseline.
        return score - base
    return (score - base) / (sota - base)


def _run_evaluator(package_dir: Path, output_root: Path) -> dict:
    """Run the package's evaluator.py with OUTPUT_DIR=output_root; return score.json."""
    evaluator = package_dir / "evaluation" / "evaluator.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"evaluator.py not found: {evaluator}")
    env = {**os.environ, "OUTPUT_DIR": str(output_root)}
    with tempfile.TemporaryDirectory(prefix="nb_eval_") as workdir:
        proc = subprocess.run(
            [sys.executable, str(evaluator)],
            cwd=workdir, env=env, capture_output=True, text=True, timeout=600,
            check=False,
        )
        score_path = Path(workdir) / "score.json"
        if not score_path.is_file():
            raise RuntimeError(
                "evaluator produced no score.json. stderr tail:\n"
                + (proc.stderr or "")[-1500:])
        return json.loads(score_path.read_text())


def _score_output(task_name: str, output_root: Path) -> dict[str, Any]:
    entry = _REGISTRY[task_name]
    package_dir = Path(entry["data_dir"])
    metadata = json.loads((package_dir / "metadata.json").read_text())
    anchors = _primary_anchors(metadata)

    raw_scores = _run_evaluator(package_dir, output_root)  # {inst: {metric: val}}

    per_instance: dict[str, float | None] = {}
    for inst, a in anchors.items():
        inst_scores = raw_scores.get(inst) or {}
        # primary metric value: the sole non-error numeric entry
        primary_val = None
        for k, v in inst_scores.items():
            if k == "error":
                continue
            try:
                primary_val = float(v)
            except (TypeError, ValueError):
                primary_val = None
            break
        per_instance[inst] = _improvement(primary_val, a["baseline"], a["sota"])

    scored = [v for v in per_instance.values() if v is not None]
    aggregate = sum(scored) / len(scored) if scored else None
    # instances that failed to score floor the aggregate toward failure
    if aggregate is not None and len(scored) < len(per_instance):
        aggregate = aggregate * len(scored) / len(per_instance)

    with _LOCK:
        prev_best = entry.get("best_aggregate")
        if aggregate is not None and (prev_best is None or aggregate > prev_best):
            entry["best_aggregate"] = aggregate
        best = entry.get("best_aggregate")

    return {
        "status": "ok",
        "aggregate_improvement": aggregate,
        "best_aggregate_improvement": best,
        "per_instance_improvement": per_instance,
        "raw_scores": raw_scores,
    }


# --------------------------------------------------------------------------- #
# Endpoints                                                                   #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/register")
def register(body: RegisterBody,
             x_naturebench_control_token: str | None = Header(default=None)) -> dict:
    data_dir = Path(body.data_dir).expanduser().resolve()
    if not (data_dir / "metadata.json").is_file():
        return JSONResponse(
            status_code=400,
            content={"status": "error",
                     "detail": f"metadata.json not found under data_dir: {data_dir}"})
    with _LOCK:
        _REGISTRY[body.task_name] = {
            "data_dir": str(data_dir),
            "out_dir": body.out_dir,
            "timeout": body.timeout,
            "started_at": None,
            "best_aggregate": None,
        }
    return {"status": "ok", "task_name": body.task_name}


@app.post("/start_timer")
def start_timer(body: TimerBody) -> dict:
    with _LOCK:
        entry = _REGISTRY.get(body.task_name)
        if entry is not None and entry.get("started_at") is None:
            entry["started_at"] = time.time()
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(body: EvaluateBody) -> dict:
    with _LOCK:
        entry = _REGISTRY.get(body.task_name)
    if entry is None:
        return JSONResponse(status_code=404,
                            content={"status": "error",
                                     "detail": f"task not registered: {body.task_name}"})

    # The client symlinks <out_dir>/workspace -> the attempt workspace and expects
    # the service to score <out_dir>/workspace/output (not the request output_dir).
    out_dir = entry.get("out_dir")
    if out_dir:
        output_root = Path(out_dir).expanduser().resolve() / "workspace" / "output"
    else:
        output_root = Path(body.output_dir or ".").expanduser().resolve()

    try:
        return _score_output(body.task_name, output_root)
    except Exception as exc:  # noqa: BLE001 -- report as a scored failure, don't 500
        return {
            "status": "ok",
            "aggregate_improvement": None,
            "best_aggregate_improvement": entry.get("best_aggregate"),
            "per_instance_improvement": {},
            "raw_scores": {"error": f"{type(exc).__name__}: {exc}"},
        }


def _write_control_token() -> str:
    token = uuid.uuid4().hex
    token_dir = SERVICE_DIR / "eval_logs"
    token_dir.mkdir(parents=True, exist_ok=True)
    (token_dir / "eval_control_token").write_text(token)
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8321)
    args = parser.parse_args()

    global CONTROL_TOKEN
    CONTROL_TOKEN = _write_control_token()  # runner reads eval_logs/eval_control_token
    print(f"Local NatureBench-compatible eval service on "
          f"http://{args.host}:{args.port} (data-driven, no external repo)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

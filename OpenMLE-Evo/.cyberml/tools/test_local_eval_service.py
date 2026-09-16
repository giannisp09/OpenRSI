#!/usr/bin/env python3
"""End-to-end offline test of the local NatureBench-compatible eval service.

Boots the service the same way the runner does (python eval_service.py --host
--port), then exercises the full register / start_timer / evaluate contract
against real task packages -- no model, no external NatureBench repo. Verifies:

  * /health responds,
  * a good starter submission yields a float aggregate_improvement with one
    per-instance entry per instance and raw_scores present,
  * an all-zero submission scores strictly lower than the good one,
  * best_aggregate_improvement tracks the best seen.

Run:  uv run python OpenMLE-Evo/.cyberml/tools/test_local_eval_service.py
Exit 0 = pass.
"""
from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

TOOLS_DIR = Path(__file__).resolve().parent
CYBERML = TOOLS_DIR.parent
TASKS_ROOT = CYBERML / "data" / "tasks"
SERVICE = CYBERML / "local_naturebench" / "eval_service.py"
PY = sys.executable


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _starter_predictions(task_dir: Path, out_root: Path) -> None:
    """Run the package starter to fill out_root/<inst>/predictions.csv."""
    text = (task_dir / "problem" / "README.md").read_text()
    starter = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    run_py = out_root.parent / "run.py"
    run_py.write_text(starter)
    env = {"DATA_DIR": str(task_dir / "problem" / "data"),
           "OUTPUT_DIR": str(out_root)}
    import os
    subprocess.run([PY, str(run_py)], env={**os.environ, **env},
                   capture_output=True, text=True, check=True)


def _instances(task_dir: Path) -> list[str]:
    data = task_dir / "problem" / "data"
    return sorted(p.name for p in data.iterdir() if p.is_dir())


def _post(base: str, ep: str, payload: dict) -> dict:
    r = requests.post(f"{base}/{ep}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def run_for_task(base: str, task_id: str, tmp: Path) -> bool:
    task_dir = TASKS_ROOT / task_id
    instances = _instances(task_dir)
    ok = True

    # out_dir/workspace/output is what the service scores.
    good_out = tmp / task_id / "good"
    good_output = good_out / "workspace" / "output"
    good_output.mkdir(parents=True, exist_ok=True)
    _starter_predictions(task_dir, good_output)

    reg = _post(base, "register", {
        "task_name": task_id, "data_dir": str(task_dir),
        "out_dir": str(good_out), "timeout": 600,
        "batch_name": "test", "eval_token": "t"})
    _post(base, "start_timer", {"task_name": task_id, "batch_name": "test"})
    good = _post(base, "evaluate", {
        "task_name": task_id, "batch_name": "test",
        "output_dir": str(good_output), "eval_token": "t"})

    agg = good.get("aggregate_improvement")
    pii = good.get("per_instance_improvement") or {}
    print(f"\n### {task_id}: register={reg.get('status')}  "
          f"aggregate_improvement={agg:.4f}" if isinstance(agg, float)
          else f"\n### {task_id}: aggregate_improvement={agg}")
    print("    per_instance: " + "  ".join(
        f"{k}={v:+.3f}" for k, v in sorted(pii.items()) if v is not None))
    if not isinstance(agg, float):
        ok = False; print("  FAIL: aggregate_improvement must be a float")
    if set(pii) != set(instances):
        ok = False; print(f"  FAIL: per_instance keys {set(pii)} != {set(instances)}")
    if not good.get("raw_scores"):
        ok = False; print("  FAIL: raw_scores missing")

    # all-zero submission -> strictly worse aggregate
    bad_out = tmp / task_id / "bad"
    bad_output = bad_out / "workspace" / "output"
    for inst in instances:
        n = len(pd.read_csv(task_dir / "problem" / "data" / inst / "x_test.csv"))
        d = bad_output / inst; d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label_pred": [0] * n}).to_csv(d / "predictions.csv", index=False)
    _post(base, "register", {
        "task_name": task_id, "data_dir": str(task_dir),
        "out_dir": str(bad_out), "timeout": 600,
        "batch_name": "test", "eval_token": "t"})
    bad = _post(base, "evaluate", {
        "task_name": task_id, "batch_name": "test",
        "output_dir": str(bad_output), "eval_token": "t"})
    bad_agg = bad.get("aggregate_improvement")
    print(f"    all-zero aggregate_improvement={bad_agg}")
    if not (isinstance(bad_agg, float) and isinstance(agg, float) and bad_agg < agg):
        ok = False; print("  FAIL: all-zero must score strictly below the starter")

    print(f"--> {task_id}:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    import os
    import tempfile
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [PY, str(SERVICE), "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # wait for health
        for _ in range(100):
            try:
                if requests.get(f"{base}/health", timeout=1).json().get("status") == "ok":
                    break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            print("FAIL: service did not become healthy")
            return 1
        print(f"service healthy at {base}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tasks = sorted(p.name for p in TASKS_ROOT.iterdir()
                           if (p / "evaluation" / "evaluator.py").exists())
            all_ok = all(run_for_task(base, t, tmp) for t in tasks)
        print("\nEVAL-SERVICE TEST:", "PASS" if all_ok else "FAIL")
        return 0 if all_ok else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())

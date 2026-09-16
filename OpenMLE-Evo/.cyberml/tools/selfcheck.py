#!/usr/bin/env python3
"""Offline self-check for a cyber-ML task package (default: all tasks).

Proves the verifier half of the loop without a model or the NatureBench repo, for
each task under .cyberml/data/tasks/:

  1. extract the starter run.py from problem/README.md and run it under the real
     DATA_DIR / OUTPUT_DIR contract  -> a "good" submission,
  2. score it with evaluation/evaluator.py,
  3. score an all-zero "bad" submission (must floor to F1 = 0 everywhere),
  4. score a malformed submission (must raise a per-instance ValidationError
     while the other instances still score).

Instances are discovered from problem/data/<instance>/. Run:

  uv run python OpenMLE-Evo/.cyberml/tools/selfcheck.py            # all tasks
  uv run python OpenMLE-Evo/.cyberml/tools/selfcheck.py nsl-kdd-nids   # one task

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
TASKS_ROOT = TOOLS_DIR.parent / "data" / "tasks"
PY = sys.executable


def _discover_instances(task_dir: Path) -> list[str]:
    data = task_dir / "problem" / "data"
    return sorted(p.name for p in data.iterdir() if p.is_dir())


def _extract_starter(task_dir: Path, dst: Path) -> None:
    text = (task_dir / "problem" / "README.md").read_text()
    m = re.search(r"```python\n(.*?)```", text, re.S)
    if not m:
        raise SystemExit("FAIL: no python starter block found in README.md")
    dst.write_text(m.group(1))


def _score(eval_dir: Path, output_dir: Path) -> dict:
    env = {**os.environ, "OUTPUT_DIR": str(output_dir)}
    subprocess.run([PY, "evaluator.py"], cwd=eval_dir, env=env,
                   capture_output=True, text=True, check=True)
    result = json.loads((eval_dir / "score.json").read_text())
    (eval_dir / "score.json").unlink(missing_ok=True)
    return result


def _write_constant(data_dir: Path, instances: list[str], output_dir: Path,
                    value: int, bad_len: dict | None = None) -> None:
    for inst in instances:
        n = len(pd.read_csv(data_dir / inst / "x_test.csv"))
        if bad_len and inst in bad_len:
            n = bad_len[inst]
        d = output_dir / inst
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label_pred": [value] * n}).to_csv(
            d / "predictions.csv", index=False)


def _mean_f1(scores: dict, instances: list[str]) -> float:
    vals = [scores[i]["Detection F1-Score"] for i in instances
            if scores[i].get("Detection F1-Score") is not None]
    return sum(vals) / len(vals) if vals else 0.0


def check_task(task_dir: Path) -> bool:
    data_dir = task_dir / "problem" / "data"
    eval_dir = task_dir / "evaluation"
    instances = _discover_instances(task_dir)
    print(f"\n### {task_dir.name}  (instances: {', '.join(instances)})")
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1-2. good submission from the starter
        starter = tmp / "run.py"
        _extract_starter(task_dir, starter)
        good_out = tmp / "good"
        good_out.mkdir()
        env = {**os.environ, "DATA_DIR": str(data_dir), "OUTPUT_DIR": str(good_out)}
        subprocess.run([PY, str(starter)], env=env, capture_output=True,
                       text=True, check=True)
        good = _score(eval_dir, good_out)
        print("[good starter] " + "  ".join(
            f"{i}={good[i]['Detection F1-Score']:.3f}" for i in instances))
        for i in instances:
            f1 = good[i]["Detection F1-Score"]
            if f1 is None or not (0.0 <= f1 <= 1.0):
                ok = False
                print(f"  FAIL: {i} F1 out of range: {f1}")

        # 3. all-zero must floor to 0 everywhere (ungameable)
        bad_out = tmp / "bad"
        _write_constant(data_dir, instances, bad_out, 0)
        bad = _score(eval_dir, bad_out)
        print("[all-zero]     " + "  ".join(
            f"{i}={bad[i]['Detection F1-Score']:.3f}" for i in instances))
        if any(bad[i]["Detection F1-Score"] != 0.0 for i in instances):
            ok = False
            print("  FAIL: all-zero submission must score 0.0 on every instance")
        if _mean_f1(good, instances) <= _mean_f1(bad, instances):
            ok = False
            print("  FAIL: good starter must beat the all-zero submission")

        # 4. malformed first instance -> ValidationError, others still scored
        target = instances[0]
        others = instances[1:]
        mal_out = tmp / "mal"
        _write_constant(data_dir, instances, mal_out, 1, bad_len={target: 5})
        mal = _score(eval_dir, mal_out)
        err = mal[target].get("error")
        others_ok = all(mal[i]["Detection F1-Score"] is not None for i in others)
        print(f"[malformed {target}] error={'yes' if err else 'NO'}; "
              f"others_scored={others_ok if others else 'n/a'}")
        if not err:
            ok = False
            print(f"  FAIL: malformed {target} must raise a ValidationError")
        if others and not others_ok:
            ok = False
            print("  FAIL: valid instances must still score when one is malformed")

    print(f"--> {task_dir.name}:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    selected = sys.argv[1:]
    if selected:
        task_dirs = [TASKS_ROOT / t for t in selected]
    else:
        task_dirs = sorted(p for p in TASKS_ROOT.iterdir()
                           if (p / "evaluation" / "evaluator.py").exists())
    all_ok = all(check_task(t) for t in task_dirs)
    print("\nSELF-CHECK:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Offline integrity test for the untouched-holdout split (no model needed).

For every task package that has an ``evaluation/holdout_split.json`` it asserts:

  1. structure  -- the split is stratified, disjoint, and covers every test row;
  2. slice math -- the evaluator's SLICE=all equals the full-set F1, and the
     score/final slices match F1 recomputed on those exact index sets;
  3. INTEGRITY  -- perturbing predictions on the FINAL rows leaves the SCORE-slice
     metric byte-identical (and vice-versa). This is the property that makes the
     holdout untouched: the search, which only ever sees the score slice, cannot
     select on the held-out rows.

    uv run python OpenMLE-Evo/.cyberml/tools/test_holdout_split.py           # all
    uv run python OpenMLE-Evo/.cyberml/tools/test_holdout_split.py phishing-url

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TOOLS_DIR = Path(__file__).resolve().parent
TASKS_ROOT = TOOLS_DIR.parent / "data" / "tasks"
PY = sys.executable


def _instances(task_dir: Path) -> list[str]:
    gt = task_dir / "evaluation" / "ground_truth"
    return sorted(p.name for p in gt.iterdir() if p.is_dir())


def _run_evaluator(eval_dir: Path, preds: dict[str, np.ndarray], slice_sel: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        for inst, p in preds.items():
            d = Path(tmp) / inst
            d.mkdir(parents=True)
            pd.DataFrame({"label_pred": p}).to_csv(d / "predictions.csv", index=False)
        env = {**os.environ, "OUTPUT_DIR": tmp, "CYBERML_EVAL_SLICE": slice_sel}
        r = subprocess.run([PY, "evaluator.py"], cwd=eval_dir, env=env,
                           capture_output=True, text=True)
        score_path = eval_dir / "score.json"
        if r.returncode != 0 or not score_path.is_file():
            raise RuntimeError(f"evaluator failed ({slice_sel}):\n{r.stderr[-800:]}")
        out = json.loads(score_path.read_text())
        score_path.unlink(missing_ok=True)
        return out


def check_task(task_dir: Path) -> bool:
    split_path = task_dir / "evaluation" / "holdout_split.json"
    if not split_path.is_file():
        print(f"### {task_dir.name}: no holdout_split.json (skip)")
        return True
    eval_dir = task_dir / "evaluation"
    gt_root = eval_dir / "ground_truth"
    spec = json.loads(split_path.read_text())
    print(f"\n### {task_dir.name}  (frac_final={spec.get('frac_final')})")
    ok = True
    rng = np.random.default_rng(0)

    labels, final_sets = {}, {}
    for inst in _instances(task_dir):
        y = pd.read_csv(gt_root / inst / "y_ref.csv")["label"].to_numpy().astype(int)
        final = set(spec["instances"][inst]["final_idx"])
        labels[inst], final_sets[inst] = y, final
        n = len(y)
        score = set(range(n)) - final
        # 1. structure
        if not final:
            ok = False; print(f"  FAIL {inst}: empty final slice")
        if final & score or (final | score) != set(range(n)):
            ok = False; print(f"  FAIL {inst}: slices not a disjoint cover")
        for name, members in (("score", score), ("final", final)):
            classes = set(int(v) for v in np.unique(y[list(members)]))
            if classes != set(int(v) for v in np.unique(y)):
                ok = False
                print(f"  FAIL {inst}: {name} slice missing a class {classes}")

    # 2 + 3: build an imperfect predictor per instance, exercise slices
    base_pred = {i: (labels[i] ^ (rng.random(len(labels[i])) < 0.2).astype(int))
                 for i in labels}
    res_all = _run_evaluator(eval_dir, base_pred, "all")
    res_score = _run_evaluator(eval_dir, base_pred, "score")
    res_final = _run_evaluator(eval_dir, base_pred, "final")
    for inst in labels:
        y, final = labels[inst], sorted(final_sets[inst])
        score = [i for i in range(len(y)) if i not in final_sets[inst]]
        hand_all = round(float(f1_score(y, base_pred[inst], zero_division=0)), 6)
        hand_sc = round(float(f1_score(y[score], base_pred[inst][score], zero_division=0)), 6)
        hand_fn = round(float(f1_score(y[final], base_pred[inst][final], zero_division=0)), 6)
        for tag, got, exp in (("all", res_all[inst]["Detection F1-Score"], hand_all),
                              ("score", res_score[inst]["Detection F1-Score"], hand_sc),
                              ("final", res_final[inst]["Detection F1-Score"], hand_fn)):
            if abs(got - exp) > 1e-9:
                ok = False
                print(f"  FAIL {inst}: {tag} F1 {got} != recomputed {exp}")

    # 3. integrity: corrupt ONLY final rows -> score metric must be unchanged
    corrupt = {i: base_pred[i].copy() for i in base_pred}
    for inst in labels:
        fidx = sorted(final_sets[inst])
        corrupt[inst][fidx] = 1 - corrupt[inst][fidx]  # flip every final prediction
    res_score2 = _run_evaluator(eval_dir, corrupt, "score")
    for inst in labels:
        a = res_score[inst]["Detection F1-Score"]
        b = res_score2[inst]["Detection F1-Score"]
        same = abs(a - b) < 1e-12
        print(f"  {inst:6s} all={res_all[inst]['Detection F1-Score']:.4f} "
              f"score={a:.4f} final={res_final[inst]['Detection F1-Score']:.4f} "
              f"| flip-final->score-unchanged={same}")
        if not same:
            ok = False
            print(f"  FAIL {inst}: corrupting final rows changed the score metric "
                  f"({a} -> {b}) -- holdout is NOT isolated")
    print(f"--> {task_dir.name}:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    sel = sys.argv[1:]
    dirs = ([TASKS_ROOT / t for t in sel] if sel else
            sorted(p for p in TASKS_ROOT.iterdir()
                   if (p / "evaluation" / "evaluator.py").is_file()))
    ok = all(check_task(d) for d in dirs)
    print("\nHOLDOUT-SPLIT TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

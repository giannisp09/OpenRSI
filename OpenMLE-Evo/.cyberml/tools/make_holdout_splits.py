#!/usr/bin/env python3
"""Generate a committed, deterministic *untouched holdout* split per task package.

Why: the evo search selects the winning program by the F1 the verifier returns
each node. Over N nodes that is N adaptive queries against one fixed hidden test
set -- classic adaptive overfitting. To make "closed X% of the gap" defensible we
carve each task's hidden test set into two disjoint, stratified slices:

  * SCORE  slice  -> the live evaluator scores this; the search selects on it.
  * FINAL  slice  -> never enters the metric the search sees; scored exactly once,
                     offline, on the winning program by tools/score_holdout.py.

The split is a stratified subset of row indices into the existing
``x_test.csv`` / ``y_ref.csv`` (their shared row order), so nothing about the data
files or the candidate contract changes: candidates still predict every test row;
the evaluator simply restricts the F1 it *returns* to the requested slice
(``CYBERML_EVAL_SLICE``, default ``score``). Absent this file the evaluator scores
all rows -- i.e. exactly today's behaviour -- so generating splits is opt-in and
runs made before it stay reproducible.

Writes ``evaluation/holdout_split.json`` into each task package. Deterministic:
same seed + fraction -> identical indices.

    python tools/make_holdout_splits.py                 # all tasks, 30% holdout
    python tools/make_holdout_splits.py --frac-final 0.25 --task nsl-kdd-nids
    python tools/make_holdout_splits.py --dry-run       # print sizes, write nothing
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
TASKS_ROOT = TOOLS_DIR.parent / "data" / "tasks"
LABEL_COL = "label"
SPLIT_FILENAME = "holdout_split.json"
DEFAULT_SEED = 20260917
DEFAULT_FRAC = 0.30


def _instances(task_dir: Path) -> list[str]:
    gt = task_dir / "evaluation" / "ground_truth"
    return sorted(p.name for p in gt.iterdir() if p.is_dir()) if gt.is_dir() else []


def _stratified_final_idx(labels: np.ndarray, frac: float, seed: int) -> list[int]:
    """Row indices assigned to the FINAL (untouched) slice.

    Stratified: takes ``frac`` of each class, so both slices keep the class
    balance. Guarantees, whenever a class has >= 2 members, that at least one of
    its members lands on each side (so neither slice loses a class it should
    have and F1's positive class is always defined)."""
    rng = np.random.default_rng(seed)
    final: list[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * frac))
        if len(idx) >= 2:
            k = min(max(k, 1), len(idx) - 1)  # keep >=1 on each side
        else:
            k = 0  # singleton class: keep it on the score side
        final.extend(int(i) for i in idx[:k])
    return sorted(final)


def build_split(task_dir: Path, frac: float, seed: int) -> dict:
    gt_root = task_dir / "evaluation" / "ground_truth"
    instances: dict[str, dict] = {}
    for inst in _instances(task_dir):
        y = pd.read_csv(gt_root / inst / "y_ref.csv")[LABEL_COL].to_numpy().astype(int)
        final_idx = _stratified_final_idx(y, frac, seed)
        final_mask = np.zeros(len(y), dtype=bool)
        final_mask[final_idx] = True
        instances[inst] = {
            "n": int(len(y)),
            "n_final": int(final_mask.sum()),
            "n_score": int((~final_mask).sum()),
            "final_pos": int(y[final_mask].sum()),
            "final_neg": int((y[final_mask] == 0).sum()),
            "score_pos": int(y[~final_mask].sum()),
            "score_neg": int((y[~final_mask] == 0).sum()),
            "final_idx": final_idx,
        }
    return {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "frac_final": frac,
        "label_col": LABEL_COL,
        "note": (
            "final_idx: 0-based row indices into x_test.csv / y_ref.csv (shared "
            "order) reserved for the untouched holdout. The live evaluator scores "
            "the COMPLEMENT (selection slice) unless CYBERML_EVAL_SLICE=final|all. "
            "Regenerate with tools/make_holdout_splits.py (deterministic)."
        ),
        "instances": instances,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", default=[],
                    help="task id (repeatable); default = every package")
    ap.add_argument("--frac-final", type=float, default=DEFAULT_FRAC,
                    help=f"fraction of each class held out (default {DEFAULT_FRAC})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the split sizes but do not write files")
    args = ap.parse_args()
    if not (0.0 < args.frac_final < 1.0):
        ap.error("--frac-final must be in (0, 1)")

    tasks = args.task or sorted(
        p.name for p in TASKS_ROOT.iterdir()
        if (p / "evaluation" / "evaluator.py").is_file())

    print(f"{'task/instance':30s} {'n':>7} {'score':>7} {'final':>7} "
          f"{'score_pos':>9} {'final_pos':>9}")
    written = 0
    for task_id in tasks:
        task_dir = TASKS_ROOT / task_id
        if not (task_dir / "evaluation" / "evaluator.py").is_file():
            print(f"  skip {task_id}: no evaluator.py")
            continue
        split = build_split(task_dir, args.frac_final, args.seed)
        for inst, s in split["instances"].items():
            print(f"{task_id + '/' + inst:30s} {s['n']:7d} {s['n_score']:7d} "
                  f"{s['n_final']:7d} {s['score_pos']:9d} {s['final_pos']:9d}")
        if not args.dry_run:
            out = task_dir / "evaluation" / SPLIT_FILENAME
            out.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
            written += 1
    if args.dry_run:
        print("\n(dry run: no files written)")
    else:
        print(f"\nWrote evaluation/{SPLIT_FILENAME} for {written} task(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

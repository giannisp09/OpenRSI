#!/usr/bin/env python3
"""Score a finished run's winning program on the UNTOUCHED holdout slice.

The evo search selected its winner using only the *score* slice (the live
evaluator hides the *final* slice -- see tools/make_holdout_splits.py and
tools/test_holdout_split.py). This tool closes the loop for the paper: it locates
the winning program in a run directory, re-executes it under the ordinary
DATA_DIR/OUTPUT_DIR candidate contract, and scores its predictions once on the
held-out rows the search never selected on -- the defensible "does it generalise,
or did it overfit the scored test set?" number.

It runs the program ``--repeats`` times (the evolved pipeline is re-fit each time)
and reports the median and IQR, so the held-out figure carries its own run-to-run
variance. It also reports the score-slice F1 of the same fits, so the
val->holdout gap (the overfitting signature) is explicit.

    uv run python OpenMLE-Evo/.cyberml/tools/score_holdout.py \
        --task nsl-kdd-nids --run-dir OpenMLE-Evo/output/<run> \
        --local-python OpenMLE-Evo/.venv/bin/python --repeats 3

Writes ``holdout_results.json`` next to the winning program (make_paper_report.py
reads it) and prints a summary. Model-written code is executed -- run it in the
same trusted/disposable environment as the search itself.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CYBERML = TOOLS_DIR.parent
TASKS_ROOT = CYBERML / "data" / "tasks"
DEFAULT_OUTPUT = CYBERML.parent / "output"
WINNER_NAMES = ("submit_code.py", "valid_code_final.py")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _anchors(task_dir: Path) -> dict[str, dict]:
    meta = json.loads((task_dir / "metadata.json").read_text())
    out = {}
    for e in meta.get("performance_entries", []):
        inst = e.get("dataset_name")
        prim = next((m for m in e.get("metrics", []) if m.get("is_primary")), None)
        if inst is None or prim is None:
            continue
        base = _num((prim.get("baseline_score") or {}).get("value"))
        sotas = prim.get("sota_score") or []
        out[inst] = {"baseline": base,
                     "sota": _num(sotas[0].get("value")) if sotas else None}
    return out


def _improvement(score, base, sota):
    if score is None:
        return None
    base = 0.0 if base is None else base
    if sota is None or sota <= base:
        return score - base
    return (score - base) / (sota - base)


def _find_winner(run_dir: Path, task: str) -> Path:
    for ep in sorted(run_dir.glob("program_ep_*")):
        for name in WINNER_NAMES:
            p = ep / task / name
            if p.is_file():
                return p
    # fallback: best step_<n>/valid_code.py by journal aggregate_improvement
    for ep in sorted(run_dir.glob("program_ep_*")):
        jr = ep / task / "aira_evo" / "checkpoint" / "journal.jsonl"
        best_step, best_agg = None, None
        if jr.is_file():
            for line in jr.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                mi = rec.get("metric_info") or {}
                agg = _num(mi.get("aggregate_improvement"))
                if agg is not None and (best_agg is None or agg > best_agg):
                    best_agg, best_step = agg, rec.get("step")
        if best_step is not None:
            cand = ep / task / f"step_{best_step}" / "valid_code.py"
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        f"No winning program found under {run_dir} for task {task} "
        f"(looked for {WINNER_NAMES} and step_<best>/valid_code.py)")


def _run_program(program: Path, data_dir: Path, python: str, seed: int,
                 timeout: float) -> Path:
    out_dir = Path(tempfile.mkdtemp(prefix="holdout_out_"))
    env = {**os.environ, "DATA_DIR": str(data_dir), "OUTPUT_DIR": str(out_dir),
           "PYTHONHASHSEED": str(seed), "CYBERML_REPLAY_SEED": str(seed)}
    r = subprocess.run([python, str(program)], env=env, capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"winning program failed (seed {seed}):\n{r.stderr[-1200:]}")
    return out_dir


def _score(eval_dir: Path, out_dir: Path, slice_sel: str, python: str) -> dict:
    env = {**os.environ, "OUTPUT_DIR": str(out_dir), "CYBERML_EVAL_SLICE": slice_sel}
    r = subprocess.run([python, "evaluator.py"], cwd=eval_dir, env=env,
                       capture_output=True, text=True)
    sp = eval_dir / "score.json"
    if r.returncode != 0 or not sp.is_file():
        raise RuntimeError(f"evaluator failed ({slice_sel}):\n{r.stderr[-800:]}")
    out = json.loads(sp.read_text())
    sp.unlink(missing_ok=True)
    return out


def _primary(inst_scores: dict):
    for k, v in (inst_scores or {}).items():
        if k == "error":
            continue
        return _num(v)
    return None


def _iqr(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--run-dir", default=None,
                    help="run directory under output/; default = latest run that "
                         "contains this task")
    ap.add_argument("--local-python", default=sys.executable,
                    help="interpreter for the candidate program (default: this one)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--program", default=None, help="override the winner path")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    # Make the interpreter path absolute: _score runs it with cwd=eval_dir, where a
    # relative ".venv/bin/python" would not resolve. Use abspath, NOT resolve():
    # a venv's bin/python is a symlink to the base interpreter, and following it
    # (resolve) would bypass the venv's site-packages. abspath keeps the venv.
    lp = args.local_python
    if "/" in lp or Path(lp).exists():
        lp = os.path.abspath(os.path.expanduser(lp))
    args.local_python = lp

    task_dir = TASKS_ROOT / args.task
    if not (task_dir / "metadata.json").is_file():
        ap.error(f"unknown task: {args.task}")
    if not (task_dir / "evaluation" / "holdout_split.json").is_file():
        ap.error(f"{args.task} has no holdout_split.json; run make_holdout_splits.py first")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        cands = [d for d in Path(args.output_dir).glob("*")
                 if any((d).glob(f"program_ep_*/{args.task}"))]
        if not cands:
            ap.error(f"no run under {args.output_dir} contains task {args.task}")
        run_dir = max(cands, key=lambda d: d.stat().st_mtime)

    program = Path(args.program).resolve() if args.program else _find_winner(run_dir, args.task)
    data_dir = task_dir / "problem" / "data"
    eval_dir = task_dir / "evaluation"
    anchors = _anchors(task_dir)
    print(f"task={args.task}\nrun={run_dir}\nwinner={program}\nrepeats={args.repeats}\n")

    per_seed = []  # list of {inst: {final, score}}
    for r in range(args.repeats):
        seed = 1000 + r
        out_dir = _run_program(program, data_dir, args.local_python, seed, args.timeout)
        try:
            fin = _score(eval_dir, out_dir, "final", args.local_python)
            sco = _score(eval_dir, out_dir, "score", args.local_python)
        finally:
            subprocess.run(["rm", "-rf", str(out_dir)], check=False)
        rec = {inst: {"final": _primary(fin.get(inst)), "score": _primary(sco.get(inst))}
               for inst in anchors}
        per_seed.append(rec)
        print(f"  replay {r+1}/{args.repeats} (seed {seed}): " +
              "  ".join(f"{i} final={rec[i]['final']:.4f}" for i in anchors
                        if rec[i]['final'] is not None))

    per_instance = {}
    agg_final, agg_score = [], []
    for inst, a in anchors.items():
        fvals = [s[inst]["final"] for s in per_seed if s[inst]["final"] is not None]
        svals = [s[inst]["score"] for s in per_seed if s[inst]["score"] is not None]
        if not fvals:
            continue
        f_med, s_med = statistics.median(fvals), statistics.median(svals)
        imp_f = _improvement(f_med, a["baseline"], a["sota"])
        imp_s = _improvement(s_med, a["baseline"], a["sota"])
        per_instance[inst] = {
            "final_f1_median": round(f_med, 6), "final_f1_iqr": round(_iqr(fvals), 6),
            "score_f1_median": round(s_med, 6), "score_f1_iqr": round(_iqr(svals), 6),
            "final_norm_improvement": imp_f, "score_norm_improvement": imp_s,
            "baseline": a["baseline"], "sota": a["sota"],
            "n_repeats": len(fvals),
        }
        if imp_f is not None:
            agg_final.append(imp_f)
        if imp_s is not None:
            agg_score.append(imp_s)

    result = {
        "task": args.task,
        "run_dir": str(run_dir),
        "program": str(program),
        "repeats": args.repeats,
        "split": json.loads((eval_dir / "holdout_split.json").read_text())["frac_final"],
        "aggregate_final_improvement": (sum(agg_final) / len(agg_final)) if agg_final else None,
        "aggregate_score_improvement": (sum(agg_score) / len(agg_score)) if agg_score else None,
        "per_instance": per_instance,
    }
    result["val_to_holdout_gap"] = (
        (result["aggregate_score_improvement"] - result["aggregate_final_improvement"])
        if (result["aggregate_final_improvement"] is not None
            and result["aggregate_score_improvement"] is not None) else None)

    # write at the task's program dir (…/program_ep_*/<task>) so make_paper_report
    # finds it, even when the winner came from a step_<n>/ subdir.
    prog_task_dir = next((ep / args.task for ep in sorted(run_dir.glob("program_ep_*"))
                          if (ep / args.task).is_dir()), program.parent)
    out_path = prog_task_dir / "holdout_results.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    af, as_ = result["aggregate_final_improvement"], result["aggregate_score_improvement"]
    print(f"\n  aggregate normalized improvement:")
    print(f"    score  slice (search selected on): {as_:.4f}" if as_ is not None else "    score: n/a")
    print(f"    final  slice (untouched holdout) : {af:.4f}" if af is not None else "    final: n/a")
    if result["val_to_holdout_gap"] is not None:
        print(f"    val->holdout gap                : {result['val_to_holdout_gap']:+.4f}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

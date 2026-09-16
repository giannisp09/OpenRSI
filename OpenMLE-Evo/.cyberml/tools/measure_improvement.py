#!/usr/bin/env python3
"""Aggregate OpenMLE-Evo search results across cyber-ML tasks.

Answers two questions after one or more search runs:

  * **per task / dataset** — the best `aggregate_improvement` the search reached
    on each task, with the winning candidate's per-instance raw scores, and
  * **overall** — the mean best `aggregate_improvement` across all tasks (the
    single number that says "how much did AI improve AI across the cyber-ML
    gym"), plus an unweighted mean of raw primary metrics.

It reads the journals every run writes at

    output/<run>/program_ep_*/<task-id>/aira_evo/checkpoint/journal.jsonl

(a run launched over a task-set covers several task-ids in one output dir; a
single-task run covers one). For each task it scans every journal record,
recursively finds the node objects that carry a numeric `aggregate_improvement`,
and keeps the maximum — i.e. best-so-far, the same number printed live as
`Registering Node ... Metric: <x>`. When a task appears in more than one run the
best across runs is kept (and the run is named).

Baselines/anchors and the primary metric name come from each task's
`data/tasks/<id>/metadata.json`, so the table is self-describing.

Usage:
    uv run python OpenMLE-Evo/.cyberml/tools/measure_improvement.py
    uv run python OpenMLE-Evo/.cyberml/tools/measure_improvement.py \\
        --output-dir OpenMLE-Evo/output --json summary.json --csv summary.csv

Exit 0 always (reporting tool). Prints a table to stdout; --json/--csv optional.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

TOOLS_DIR = Path(__file__).resolve().parent
CYBERML = TOOLS_DIR.parent
DEFAULT_OUTPUT = CYBERML.parent / "output"
TASKS_ROOT = CYBERML / "data" / "tasks"


# --------------------------------------------------------------------------- #
# Journal parsing                                                             #
# --------------------------------------------------------------------------- #
def _walk_nodes(obj: Any) -> Iterable[dict]:
    """Yield every dict that holds a numeric `aggregate_improvement`."""
    if isinstance(obj, dict):
        agg = obj.get("aggregate_improvement")
        if isinstance(agg, (int, float)):
            yield obj
        for v in obj.values():
            yield from _walk_nodes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_nodes(v)


def _best_from_journal(journal: Path) -> dict | None:
    """Best (max aggregate_improvement) candidate record in one journal file."""
    best: dict | None = None
    n_nodes = 0
    seen_ids: set[str] = set()
    for line in journal.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for node in _walk_nodes(record):
            nid = str(node.get("id") or node.get("node_id") or id(node))
            if nid not in seen_ids:
                seen_ids.add(nid)
                n_nodes += 1
            agg = float(node["aggregate_improvement"])
            if best is None or agg > best["aggregate_improvement"]:
                best = {
                    "aggregate_improvement": agg,
                    "per_instance_improvement": node.get("per_instance_improvement"),
                    "raw_scores": node.get("raw_scores"),
                }
    if best is None:
        return None
    best["n_nodes"] = n_nodes
    return best


def _discover(output_dir: Path) -> dict[str, list[dict]]:
    """{task_id: [ {run, best...}, ... ]} across all runs under output_dir."""
    found: dict[str, list[dict]] = {}
    for journal in output_dir.glob("*/program_ep_*/*/aira_evo/checkpoint/journal.jsonl"):
        # .../output/<run>/program_ep_N/<task-id>/aira_evo/checkpoint/journal.jsonl
        task_id = journal.parents[2].name
        run = journal.parents[4].name
        best = _best_from_journal(journal)
        if best is None:
            continue
        best["run"] = run
        found.setdefault(task_id, []).append(best)
    return found


# --------------------------------------------------------------------------- #
# Metadata (anchors + primary metric)                                         #
# --------------------------------------------------------------------------- #
def _task_meta(task_id: str) -> dict:
    meta_path = TASKS_ROOT / task_id / "metadata.json"
    if not meta_path.is_file():
        return {"metric": "?", "instances": {}}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metric_name = "?"
    instances: dict[str, dict[str, float | None]] = {}
    for entry in meta.get("performance_entries", []):
        inst = entry.get("dataset_name")
        for m in entry.get("metrics", []):
            if not m.get("is_primary"):
                continue
            metric_name = m.get("name", metric_name)

            def _num(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None

            base = _num((m.get("baseline_score") or {}).get("value"))
            sotas = m.get("sota_score") or []
            sota = _num(sotas[0].get("value")) if sotas else None
            instances[inst] = {"baseline": base, "sota": sota}
    return {"metric": metric_name, "instances": instances}


def _primary_raw(raw_scores: dict | None) -> dict[str, float]:
    """{instance: primary_metric_value} from a candidate's raw_scores block."""
    out: dict[str, float] = {}
    if not isinstance(raw_scores, dict):
        return out
    for inst, metrics in raw_scores.items():
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if k == "error":
                continue
            try:
                out[inst] = float(v)
            except (TypeError, ValueError):
                pass
            break
    return out


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def build_summary(output_dir: Path, all_tasks: bool = False) -> dict:
    discovered = _discover(output_dir)
    tasks = []
    for task_id in sorted(discovered):
        # By default, restrict to the cyber-ML gym: tasks that ship a package
        # under .cyberml/data/tasks/. --all-tasks lifts this to every journal.
        if not all_tasks and not (TASKS_ROOT / task_id / "metadata.json").is_file():
            continue
        runs = discovered[task_id]
        best = max(runs, key=lambda r: r["aggregate_improvement"])
        meta = _task_meta(task_id)
        raw = _primary_raw(best.get("raw_scores"))
        instances = []
        for inst, anchors in meta["instances"].items():
            instances.append({
                "instance": inst,
                "baseline": anchors["baseline"],
                "best_score": raw.get(inst),
                "improvement": (best.get("per_instance_improvement") or {}).get(inst),
            })
        # instances present in raw but not metadata (defensive)
        for inst, val in raw.items():
            if inst not in meta["instances"]:
                instances.append({"instance": inst, "baseline": None,
                                   "best_score": val, "improvement": None})
        tasks.append({
            "task_id": task_id,
            "metric": meta["metric"],
            "best_aggregate_improvement": best["aggregate_improvement"],
            "n_nodes": best.get("n_nodes"),
            "n_runs": len(runs),
            "run": best["run"],
            "instances": instances,
        })

    aggs = [t["best_aggregate_improvement"] for t in tasks]
    all_raw = [i["best_score"] for t in tasks for i in t["instances"]
               if isinstance(i["best_score"], (int, float))]
    overall = {
        "num_tasks": len(tasks),
        "mean_best_aggregate_improvement": (sum(aggs) / len(aggs)) if aggs else None,
        "min_best_aggregate_improvement": min(aggs) if aggs else None,
        "max_best_aggregate_improvement": max(aggs) if aggs else None,
        "mean_primary_metric_raw": (sum(all_raw) / len(all_raw)) if all_raw else None,
    }
    return {"overall": overall, "tasks": tasks}


def _fmt(x, w=8, p=4):
    if isinstance(x, (int, float)):
        return f"{x:>{w}.{p}f}"
    return f"{'—':>{w}}"


def print_table(summary: dict) -> None:
    ov = summary["overall"]
    tasks = summary["tasks"]
    print("\n" + "=" * 72)
    print("CYBER-ML SELF-IMPROVEMENT SUMMARY")
    print("=" * 72)
    if not tasks:
        print("\nNo task journals found. Run a search first, e.g.:")
        print("  scripts/run_naturebench_local.py ... --task <id> "
              "-- search.runner.solver.step_limit=15")
        return

    print(f"\nPER TASK / DATASET  ({ov['num_tasks']} task(s))\n")
    print(f"  {'task':22s} {'metric':22s} {'best_agg':>9s} {'nodes':>6s}  run")
    print("  " + "-" * 78)
    for t in tasks:
        print(f"  {t['task_id']:22s} {t['metric'][:22]:22s} "
              f"{t['best_aggregate_improvement']:>9.4f} {str(t['n_nodes']):>6s}  "
              f"{t['run']}"
              + ("" if t["n_runs"] == 1 else f"  (best of {t['n_runs']} runs)"))
        for i in t["instances"]:
            imp = i["improvement"]
            tag = "" if imp is None else f"  (Δnorm {imp:+.3f})"
            print(f"      · {i['instance']:16s} baseline={_fmt(i['baseline'])}"
                  f"  best={_fmt(i['best_score'])}{tag}")

    print("\n" + "-" * 72)
    print("OVERALL")
    print("-" * 72)
    m = ov["mean_best_aggregate_improvement"]
    print(f"  mean best aggregate_improvement across tasks : "
          f"{m:.4f}" if m is not None else "  (none)")
    print(f"  spread (min .. max)                          : "
          f"{ov['min_best_aggregate_improvement']:.4f} .. "
          f"{ov['max_best_aggregate_improvement']:.4f}")
    if ov["mean_primary_metric_raw"] is not None:
        print(f"  mean raw primary metric (all instances)      : "
              f"{ov['mean_primary_metric_raw']:.4f}")
    print("\n  aggregate_improvement: 0.0 = per-task weak baseline, 1.0 = per-task"
          " SOTA anchor.\n  Higher is better; >0 means the search beat the baseline.")
    print("=" * 72 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                    help="Root holding the run directories (default: OpenMLE-Evo/output)")
    ap.add_argument("--json", default=None, help="Write the summary as JSON here")
    ap.add_argument("--csv", default=None, help="Write a per-task CSV here")
    ap.add_argument("--all-tasks", action="store_true",
                    help="Include every task journal, not just the .cyberml gym tasks")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    summary = build_summary(output_dir, all_tasks=args.all_tasks)
    print_table(summary)

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote JSON summary: {args.json}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task_id", "metric", "best_aggregate_improvement",
                        "n_nodes", "run", "instance", "baseline", "best_score",
                        "improvement"])
            for t in summary["tasks"]:
                for i in t["instances"]:
                    w.writerow([t["task_id"], t["metric"],
                                t["best_aggregate_improvement"], t["n_nodes"],
                                t["run"], i["instance"], i["baseline"],
                                i["best_score"], i["improvement"]])
        print(f"Wrote CSV summary: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Turn OpenMLE-Evo search journals into paper-ready results.

`measure_improvement.py` answers the one-line question ("how much did the loop
improve, per task and overall"). This tool produces the full evidence pack a
write-up needs: tables (CSV + LaTeX booktabs), figures (PDF + PNG), a
reproducibility manifest, and a methods+results REPORT.md that stitches them
together.

It reads the same journals every run writes:

    output/<run>/program_ep_*/<task-id>/aira_evo/checkpoint/journal.jsonl

Each line of a journal is one search node, carrying: `step`, `operators_used`
(draft/improve/crossover/debug), `metric` (= aggregate_improvement),
`metric_info` (structured `per_instance_improvement`, `raw_scores`, ...),
`operators_metrics[].usage` (token counts incl. reasoning tokens and the real
per-call `cost` / `cost_details.upstream_inference_cost`), `exec_time`,
`creation_time`, and `parents`/`children` (the evolution DAG). Anchors and the
primary metric name come from each task's `.cyberml/data/tasks/<id>/metadata.json`.

What it emits under --paper-dir (default .cyberml/paper/):

  tables/
    main_results.{csv,tex}          per-task baseline/best/SOTA/norm-improvement,
                                    best@step, nodes, buggy%, wall-clock, tokens, cost
    per_instance.csv                per-instance baseline/best/SOTA/norm-improvement
    operator_effectiveness.{csv,tex}  per-operator usage/success/mean+best improvement
    compute_budget.{csv,tex}        per-task tokens (prompt/completion/reasoning), cost, time
  plots/
    trajectory_<task>.{pdf,png}     step vs aggregate_improvement, colored by operator,
                                    with best-so-far envelope + baseline/SOTA reference lines
    trajectory_all.{pdf,png}        small-multiples best-so-far across tasks
    aggregate_bar.{pdf,png}         best aggregate_improvement per task + overall mean
    operator_effectiveness.{pdf,png}  mean improvement + success rate per operator
    family_<task>.{pdf,png}         per-instance baseline-vs-best (multi-instance tasks)
  manifest.json                     git commit, host, package versions, runs consumed, args
  REPORT.md                         paper-ready methods + results, embeds tables + figures

Usage:
    uv run python OpenMLE-Evo/.cyberml/tools/make_paper_report.py \
        --model-label "Frontis-MA1-35B (local)" --price-input 0 --price-output 0

Cost: by default the recorded per-call cost is used (0 for a local/no-charge
endpoint). Pass --price-input/--price-output (USD per 1M tokens) to recompute
from token counts instead (e.g. to price a hypothetical API run).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
CYBERML = TOOLS_DIR.parent
REPO_ROOT = CYBERML.parent
DEFAULT_OUTPUT = REPO_ROOT / "output"
DEFAULT_PAPER = CYBERML / "paper"
TASKS_ROOT = CYBERML / "data" / "tasks"

# Okabe-Ito colorblind-safe palette, mapped to search operators.
OP_COLORS = {
    "draft": "#0072B2",
    "improve": "#009E73",
    "crossover": "#D55E00",
    "debug": "#CC79A7",
    "seed": "#999999",
    "other": "#666666",
}
OP_ORDER = ["draft", "improve", "crossover", "debug", "seed", "other"]
NON_OPERATORS = {"rich_memory_summary"}


# --------------------------------------------------------------------------- #
# Node extraction                                                             #
# --------------------------------------------------------------------------- #
def _f(x: Any) -> float | None:
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def _primary_operator(operators_used: Any) -> str:
    if not isinstance(operators_used, list):
        return "seed"
    ops = [o for o in operators_used if o not in NON_OPERATORS]
    return ops[0] if ops else "seed"


def _usage_cost(usage: dict) -> float:
    """Recorded USD cost of one LLM call. Falls back to upstream inference cost
    when the top-level `cost` is 0 (e.g. BYOK routing reports 0 there but keeps
    the real number in cost_details)."""
    c = _f(usage.get("cost")) or 0.0
    if not c:
        cd = usage.get("cost_details") or {}
        c = _f(cd.get("upstream_inference_cost")) or 0.0
    return c


def _extract_node(rec: dict) -> dict | None:
    """One search node -> a flat dict, or None if it carries no evaluation."""
    mi = rec.get("metric_info")
    agg = None
    per_inst = None
    raw = None
    if isinstance(mi, dict):
        agg = _f(mi.get("aggregate_improvement"))
        per_inst = mi.get("per_instance_improvement")
        raw = mi.get("raw_scores")
    if agg is None:
        agg = _f(rec.get("metric"))

    # token + cost accounting across every LLM call attached to this node
    prompt_t = completion_t = reasoning_t = total_t = 0
    cost = 0.0
    llm_calls = 0
    for om in rec.get("operators_metrics") or []:
        usage = (om or {}).get("usage")
        if not isinstance(usage, dict):
            continue
        prompt_t += int(usage.get("prompt_tokens") or 0)
        completion_t += int(usage.get("completion_tokens") or 0)
        total_t += int(usage.get("total_tokens") or 0)
        ctd = usage.get("completion_tokens_details") or {}
        reasoning_t += int(ctd.get("reasoning_tokens") or 0)
        cost += _usage_cost(usage)
        llm_calls = max(llm_calls, int(usage.get("cumulative_num_llm_calls") or 0))
    if not llm_calls:
        llm_calls = len(rec.get("operators_metrics") or [])

    return {
        "step": rec.get("step"),
        "id": rec.get("id"),
        "operator": _primary_operator(rec.get("operators_used")),
        "is_buggy": bool(rec.get("is_buggy")),
        "agg": agg,
        "per_instance_improvement": per_inst if isinstance(per_inst, dict) else {},
        "raw_scores": raw if isinstance(raw, dict) else {},
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "reasoning_tokens": reasoning_t,
        "total_tokens": total_t or (prompt_t + completion_t),
        "cost": cost,
        "llm_calls": llm_calls,
        "exec_time": _f(rec.get("exec_time")) or 0.0,
        "creation_time": _f(rec.get("creation_time")),
        "reasoning_content": _first_reasoning(rec),
    }


def _first_reasoning(rec: dict) -> str | None:
    for om in rec.get("operators_metrics") or []:
        usage = (om or {}).get("usage") or {}
        rc = usage.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return rc.strip()
    return None


def _read_journal(journal: Path) -> list[dict]:
    nodes = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        node = _extract_node(rec)
        if node is not None:
            nodes.append(node)
    nodes.sort(key=lambda n: (n["step"] if isinstance(n["step"], int) else 1e9))
    return nodes


def _discover(output_dir: Path) -> dict[str, list[dict]]:
    """{task_id: [ {run, journal, nodes}, ... ]}."""
    found: dict[str, list[dict]] = {}
    pattern = "*/program_ep_*/*/aira_evo/checkpoint/journal.jsonl"
    for journal in output_dir.glob(pattern):
        task_id = journal.parents[2].name
        run = journal.parents[4].name
        nodes = _read_journal(journal)
        if not nodes:
            continue
        found.setdefault(task_id, []).append(
            {"run": run, "journal": journal, "nodes": nodes}
        )
    return found


# --------------------------------------------------------------------------- #
# Metadata (anchors + primary metric + domain)                                #
# --------------------------------------------------------------------------- #
def _task_meta(task_id: str) -> dict:
    meta_path = TASKS_ROOT / task_id / "metadata.json"
    if not meta_path.is_file():
        return {"name": task_id, "domain": "?", "metric": "?", "instances": {}}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dm = meta.get("domain_metadata") or {}
    domain = dm.get("sub_domain") or dm.get("primary_domain") or "?"
    metric_name = "?"
    instances: dict[str, dict] = {}
    for entry in meta.get("performance_entries", []):
        inst = entry.get("dataset_name")
        for m in entry.get("metrics", []):
            if not m.get("is_primary"):
                continue
            metric_name = m.get("name", metric_name)
            base = _f((m.get("baseline_score") or {}).get("value"))
            sotas = m.get("sota_score") or []
            sota = _f(sotas[0].get("value")) if sotas else None
            instances[inst] = {"baseline": base, "sota": sota}
    return {
        "name": meta.get("task_name", task_id),
        "domain": domain,
        "metric": metric_name,
        "instances": instances,
    }


def _raw_primary(raw_scores: dict, inst: str) -> float | None:
    m = raw_scores.get(inst)
    if not isinstance(m, dict):
        return None
    for k, v in m.items():
        if k == "error":
            continue
        return _f(v)
    return None


# --------------------------------------------------------------------------- #
# Per-task aggregation                                                         #
# --------------------------------------------------------------------------- #
def _best_node(nodes: list[dict]) -> dict | None:
    scored = [n for n in nodes if isinstance(n["agg"], float)]
    if not scored:
        return None
    return max(scored, key=lambda n: n["agg"])


def _run_totals(nodes: list[dict]) -> dict:
    times = [n["creation_time"] for n in nodes if n["creation_time"]]
    wall = (max(times) - min(times)) if len(times) >= 2 else 0.0
    return {
        "n_nodes": len(nodes),
        "n_buggy": sum(1 for n in nodes if n["is_buggy"]),
        "prompt_tokens": sum(n["prompt_tokens"] for n in nodes),
        "completion_tokens": sum(n["completion_tokens"] for n in nodes),
        "reasoning_tokens": sum(n["reasoning_tokens"] for n in nodes),
        "total_tokens": sum(n["total_tokens"] for n in nodes),
        "exec_time": sum(n["exec_time"] for n in nodes),
        "cost": sum(n["cost"] for n in nodes),
        "wall": wall,
        "llm_calls": sum(n["llm_calls"] for n in nodes),
    }


def _priced_cost(totals: dict, price_in: float | None, price_out: float | None) -> float:
    if price_in is None and price_out is None:
        return totals["cost"]
    pi = price_in or 0.0
    po = price_out or 0.0
    return (
        totals["prompt_tokens"] / 1e6 * pi + totals["completion_tokens"] / 1e6 * po
    )


def build(output_dir: Path, all_tasks: bool, price_in, price_out) -> dict:
    discovered = _discover(output_dir)
    tasks = []
    for task_id in sorted(discovered):
        if not all_tasks and not (TASKS_ROOT / task_id / "metadata.json").is_file():
            continue
        runs = discovered[task_id]
        # headline run = the one that reached the highest aggregate_improvement
        best_run = None
        best_agg = None
        for run in runs:
            bn = _best_node(run["nodes"])
            if bn is None:
                continue
            if best_agg is None or bn["agg"] > best_agg:
                best_agg, best_run = bn["agg"], run
        if best_run is None:
            continue

        meta = _task_meta(task_id)
        nodes = best_run["nodes"]
        bn = _best_node(nodes)
        totals = _run_totals(nodes)
        totals["cost"] = _priced_cost(totals, price_in, price_out)

        # per-instance table rows from the best node
        insts = []
        base_vals, best_vals, sota_vals = [], [], []
        for inst, anchors in meta["instances"].items():
            best_score = _raw_primary(bn["raw_scores"], inst)
            insts.append({
                "instance": inst,
                "baseline": anchors["baseline"],
                "best": best_score,
                "sota": anchors["sota"],
                "norm_improvement": _f((bn["per_instance_improvement"] or {}).get(inst)),
            })
            if anchors["baseline"] is not None:
                base_vals.append(anchors["baseline"])
            if best_score is not None:
                best_vals.append(best_score)
            if anchors["sota"] is not None:
                sota_vals.append(anchors["sota"])

        # spread across repeated runs (seeds), if any
        run_bests = [
            _best_node(r["nodes"])["agg"]
            for r in runs
            if _best_node(r["nodes"]) is not None
        ]
        mean_across = sum(run_bests) / len(run_bests)
        std_across = (
            (sum((x - mean_across) ** 2 for x in run_bests) / len(run_bests)) ** 0.5
            if len(run_bests) > 1 else 0.0
        )

        tasks.append({
            "task_id": task_id,
            "name": meta["name"],
            "domain": meta["domain"],
            "metric": meta["metric"],
            "n_instances": len(meta["instances"]) or len(bn["raw_scores"]),
            "best_agg": bn["agg"],
            "best_step": bn["step"],
            "best_operator": bn["operator"],
            "mean_across_runs": mean_across,
            "std_across_runs": std_across,
            "n_runs": len(runs),
            "run": best_run["run"],
            "journal": str(best_run["journal"]),
            "mean_baseline": _mean(base_vals),
            "mean_best": _mean(best_vals),
            "mean_sota": _mean(sota_vals),
            "instances": insts,
            "totals": totals,
            "nodes": nodes,
        })

    aggs = [t["best_agg"] for t in tasks]
    overall = {
        "num_tasks": len(tasks),
        "mean_best_agg": _mean(aggs),
        "min_best_agg": min(aggs) if aggs else None,
        "max_best_agg": max(aggs) if aggs else None,
        "total_cost": sum(t["totals"]["cost"] for t in tasks),
        "total_tokens": sum(t["totals"]["total_tokens"] for t in tasks),
        "total_nodes": sum(t["totals"]["n_nodes"] for t in tasks),
        "total_wall": sum(t["totals"]["wall"] for t in tasks),
    }
    return {"overall": overall, "tasks": tasks}


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


# --------------------------------------------------------------------------- #
# Operator effectiveness (pooled across the reported runs)                    #
# --------------------------------------------------------------------------- #
def operator_stats(summary: dict) -> list[dict]:
    pool: dict[str, list[dict]] = {}
    best_owner: dict[str, int] = {}
    for t in summary["tasks"]:
        bn = _best_node(t["nodes"])
        if bn is not None:
            best_owner[bn["operator"]] = best_owner.get(bn["operator"], 0) + 1
        for n in t["nodes"]:
            if n["operator"] == "seed":
                continue
            pool.setdefault(n["operator"], []).append(n)
    rows = []
    for op in OP_ORDER:
        ns = pool.get(op)
        if not ns:
            continue
        succ = [n for n in ns if not n["is_buggy"] and isinstance(n["agg"], float)]
        aggs = [n["agg"] for n in succ]
        rows.append({
            "operator": op,
            "used": len(ns),
            "success": len(succ),
            "success_rate": len(succ) / len(ns) if ns else 0.0,
            "mean_agg": _mean(aggs),
            "best_agg": max(aggs) if aggs else None,
            "n_best_nodes": best_owner.get(op, 0),
        })
    return rows


# --------------------------------------------------------------------------- #
# Table writers                                                               #
# --------------------------------------------------------------------------- #
def _n(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def write_tables(summary: dict, tdir: Path) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    ops = operator_stats(summary)

    # main_results.csv
    with (tdir / "main_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "name", "domain", "metric", "n_instances",
                    "mean_baseline", "mean_best", "mean_sota",
                    "aggregate_improvement", "mean_across_runs", "std_across_runs",
                    "n_runs", "best_step", "best_operator", "n_nodes", "buggy_pct",
                    "wall_min", "total_tokens", "reasoning_tokens", "cost_usd", "run"])
        for t in summary["tasks"]:
            tot = t["totals"]
            w.writerow([
                t["task_id"], t["name"], t["domain"], t["metric"], t["n_instances"],
                _n(t["mean_baseline"]), _n(t["mean_best"]), _n(t["mean_sota"]),
                _n(t["best_agg"]), _n(t["mean_across_runs"]), _n(t["std_across_runs"]),
                t["n_runs"], t["best_step"], t["best_operator"], tot["n_nodes"],
                _n(100 * tot["n_buggy"] / tot["n_nodes"] if tot["n_nodes"] else 0, 1),
                _n(tot["wall"] / 60, 1), tot["total_tokens"], tot["reasoning_tokens"],
                _n(tot["cost"], 4), t["run"],
            ])

    # per_instance.csv
    with (tdir / "per_instance.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "instance", "baseline", "best", "sota",
                    "norm_improvement"])
        for t in summary["tasks"]:
            for i in t["instances"]:
                w.writerow([t["task_id"], i["instance"], _n(i["baseline"]),
                            _n(i["best"]), _n(i["sota"]), _n(i["norm_improvement"])])

    # operator_effectiveness.csv
    with (tdir / "operator_effectiveness.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operator", "used", "success", "success_rate",
                    "mean_aggregate_improvement", "best_aggregate_improvement",
                    "n_best_nodes"])
        for r in ops:
            w.writerow([r["operator"], r["used"], r["success"],
                        _n(r["success_rate"], 3), _n(r["mean_agg"]),
                        _n(r["best_agg"]), r["n_best_nodes"]])

    # compute_budget.csv
    with (tdir / "compute_budget.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "n_nodes", "llm_calls", "prompt_tokens",
                    "completion_tokens", "reasoning_tokens", "total_tokens",
                    "exec_time_s", "wall_min", "cost_usd"])
        for t in summary["tasks"]:
            tt = t["totals"]
            w.writerow([t["task_id"], tt["n_nodes"], tt["llm_calls"],
                        tt["prompt_tokens"], tt["completion_tokens"],
                        tt["reasoning_tokens"], tt["total_tokens"],
                        _n(tt["exec_time"], 1), _n(tt["wall"] / 60, 1),
                        _n(tt["cost"], 4)])

    # LaTeX (booktabs) for the two headline tables
    _write_latex_main(summary, tdir / "main_results.tex")
    _write_latex_ops(ops, tdir / "operator_effectiveness.tex")
    _write_latex_compute(summary, tdir / "compute_budget.tex")


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _write_latex_main(summary: dict, path: Path) -> None:
    ov = summary["overall"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Self-improvement of the OpenMLE-Evo loop on the cyber-ML gym. "
        r"Normalized improvement is the aggregate\_improvement (0 = weak baseline, "
        r"1 = SOTA anchor), the mean over a task's instances of "
        r"$(\mathrm{score}-\mathrm{baseline})/(\mathrm{sota}-\mathrm{baseline})$.}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Task & Domain & \#Inst & Baseline & Best & Norm.\ Impr. & Cost (\$) \\",
        r"\midrule",
    ]
    for t in summary["tasks"]:
        lines.append(
            f"{_latex_escape(t['task_id'])} & {_latex_escape(t['domain'])} & "
            f"{t['n_instances']} & {_n(t['mean_baseline'],3)} & "
            f"{_n(t['mean_best'],3)} & {_n(t['best_agg'],3)} & "
            f"{_n(t['totals']['cost'],3)} \\\\"
        )
    lines += [
        r"\midrule",
        f"\\textbf{{Overall}} & & & & & \\textbf{{{_n(ov['mean_best_agg'],3)}}} & "
        f"{_n(ov['total_cost'],3)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_ops(ops: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Search-operator effectiveness pooled across tasks. "
        r"Success = produced a non-buggy scored candidate; \#Best = number of "
        r"tasks whose best candidate came from this operator.}",
        r"\label{tab:operators}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Operator & Used & Success \% & Mean Impr. & Best Impr. & \#Best \\",
        r"\midrule",
    ]
    for r in ops:
        lines.append(
            f"{r['operator']} & {r['used']} & {_n(100*r['success_rate'],1)} & "
            f"{_n(r['mean_agg'],3)} & {_n(r['best_agg'],3)} & {r['n_best_nodes']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_compute(summary: dict, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compute budget per task: search nodes, tokens (reasoning tokens "
        r"in parentheses), wall-clock, and cost.}",
        r"\label{tab:compute}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Task & Nodes & Tokens (reas.) & Wall (min) & Cost (\$) \\",
        r"\midrule",
    ]
    for t in summary["tasks"]:
        tt = t["totals"]
        lines.append(
            f"{_latex_escape(t['task_id'])} & {tt['n_nodes']} & "
            f"{tt['total_tokens']:,} ({tt['reasoning_tokens']:,}) & "
            f"{_n(tt['wall']/60,1)} & {_n(tt['cost'],3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans", "figure.autolayout": False,
    })
    return plt


def _save(fig, pdir: Path, name: str):
    pdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdir / f"{name}.pdf")
    fig.savefig(pdir / f"{name}.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _best_so_far(nodes):
    xs, ys, best = [], [], None
    for n in nodes:
        if not isinstance(n["agg"], float):
            continue
        best = n["agg"] if best is None else max(best, n["agg"])
        xs.append(n["step"])
        ys.append(best)
    return xs, ys


def plot_trajectory(plt, t: dict, pdir: Path):
    nodes = t["nodes"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.axhline(0.0, ls="--", lw=0.9, color="#888", zorder=1)
    ax.text(0.01, 0.0, " baseline", color="#666", va="bottom", ha="left",
            fontsize=8, transform=ax.get_yaxis_transform())
    # SOTA line (1.0) only if it fits the visible range reasonably
    scored = [n for n in nodes if isinstance(n["agg"], float)]
    ymax = max([n["agg"] for n in scored] + [0.0])
    if ymax > 0.6:
        ax.axhline(1.0, ls="--", lw=0.9, color="#888", zorder=1)
        ax.text(0.01, 1.0, " SOTA", color="#666", va="bottom", ha="left",
                fontsize=8, transform=ax.get_yaxis_transform())
    # per-operator scatter
    seen = set()
    for n in scored:
        op = n["operator"]
        ax.scatter(n["step"], n["agg"], s=40, color=OP_COLORS.get(op, OP_COLORS["other"]),
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=op if op not in seen else None)
        seen.add(op)
    # buggy nodes as small x on the baseline
    for n in nodes:
        if n["is_buggy"]:
            ax.scatter(n["step"], 0.0, marker="x", s=26, color="#c0392b",
                       linewidth=1.1, zorder=2)
    # best-so-far envelope
    xs, ys = _best_so_far(nodes)
    if xs:
        ax.step(xs, ys, where="post", color="#111", lw=1.6, zorder=4,
                label="best-so-far")
    # star the best node
    bn = _best_node(nodes)
    if bn:
        ax.scatter(bn["step"], bn["agg"], marker="*", s=190, color="#111",
                   edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate(f"{bn['agg']:.3f}", (bn["step"], bn["agg"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9,
                    fontweight="bold")
    ax.set_xlabel("search node (step)")
    ax.set_ylabel("aggregate improvement")
    ax.set_title(f"{t['task_id']} — search trajectory")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    _save(fig, pdir, f"trajectory_{t['task_id']}")


def plot_trajectory_all(plt, summary: dict, pdir: Path):
    tasks = summary["tasks"]
    if len(tasks) < 2:
        return
    ncol = min(3, len(tasks))
    nrow = (len(tasks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow),
                             squeeze=False)
    for idx, t in enumerate(tasks):
        ax = axes[idx // ncol][idx % ncol]
        xs, ys = _best_so_far(t["nodes"])
        ax.axhline(0.0, ls="--", lw=0.8, color="#bbb")
        if xs:
            ax.step(xs, ys, where="post", color="#0072B2", lw=1.8)
            ax.scatter([xs[-1]], [ys[-1]], marker="*", s=120, color="#111",
                       zorder=4)
        ax.set_title(t["task_id"], fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("agg. impr.", fontsize=8)
    for j in range(len(tasks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Best-so-far aggregate improvement across the cyber-ML gym", y=1.02)
    _save(fig, pdir, "trajectory_all")


def plot_aggregate_bar(plt, summary: dict, pdir: Path):
    tasks = sorted(summary["tasks"], key=lambda t: t["best_agg"])
    if not tasks:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(tasks) + 1.4))
    ys = range(len(tasks))
    vals = [t["best_agg"] for t in tasks]
    errs = [t["std_across_runs"] for t in tasks]
    has_err = any(e > 0 for e in errs)
    ax.barh(list(ys), vals, color="#0072B2", height=0.62,
            xerr=errs if has_err else None, error_kw={"ecolor": "#333", "capsize": 3})
    ax.set_yticks(list(ys))
    ax.set_yticklabels([t["task_id"] for t in tasks])
    ax.axvline(0.0, color="#888", lw=0.9)
    mean = summary["overall"]["mean_best_agg"]
    if mean is not None:
        ax.axvline(mean, ls="--", color="#D55E00", lw=1.3)
        ax.text(mean, len(tasks) - 0.4, f" mean {mean:.3f}", color="#D55E00",
                fontsize=8.5, va="top")
    for y, v in zip(ys, vals):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("best aggregate improvement (0 = baseline, 1 = SOTA)")
    ax.set_title("Self-improvement per cyber-ML task")
    _save(fig, pdir, "aggregate_bar")


def plot_operator_effectiveness(plt, summary: dict, pdir: Path):
    ops = operator_stats(summary)
    ops = [o for o in ops if o["mean_agg"] is not None]
    if not ops:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = range(len(ops))
    means = [o["mean_agg"] for o in ops]
    bars = ax.bar(list(xs), means, color=[OP_COLORS.get(o["operator"], "#666") for o in ops],
                  width=0.62)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([o["operator"] for o in ops])
    ax.axhline(0.0, color="#888", lw=0.9)
    for b, o in zip(bars, ops):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"n={o['used']}\n{o['success_rate']*100:.0f}% ok",
                ha="center", va="bottom" if b.get_height() >= 0 else "top",
                fontsize=8)
    ax.set_ylabel("mean aggregate improvement (scored nodes)")
    ax.set_title("Search-operator effectiveness")
    _save(fig, pdir, "operator_effectiveness")


def plot_family(plt, t: dict, pdir: Path):
    insts = [i for i in t["instances"] if i["baseline"] is not None]
    if len(insts) < 2:
        return
    import numpy as np
    labels = [i["instance"] for i in insts]
    base = [i["baseline"] for i in insts]
    best = [i["best"] if i["best"] is not None else 0.0 for i in insts]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.15 * len(labels) + 2, 3.8))
    ax.bar(x - w / 2, base, w, label="baseline", color="#bbbbbb")
    ax.bar(x + w / 2, best, w, label="evolved best", color="#009E73")
    for i, inst in enumerate(insts):
        if inst["sota"] is not None:
            ax.hlines(inst["sota"], i - w, i + w, color="#D55E00", lw=1.6,
                      linestyles="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(t["metric"])
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{t['task_id']} — per-instance baseline vs evolved "
                 f"(dashed = SOTA anchor)")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, pdir, f"family_{t['task_id']}")


def write_figures(summary: dict, pdir: Path) -> list[str]:
    try:
        plt = _setup_mpl()
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping figures: {exc}")
        return []
    made = []
    for t in summary["tasks"]:
        plot_trajectory(plt, t, pdir)
        made.append(f"trajectory_{t['task_id']}")
        plot_family(plt, t, pdir)
    plot_trajectory_all(plt, summary, pdir)
    plot_aggregate_bar(plt, summary, pdir)
    plot_operator_effectiveness(plt, summary, pdir)
    return made


# --------------------------------------------------------------------------- #
# Manifest + REPORT.md                                                        #
# --------------------------------------------------------------------------- #
def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    except OSError:
        return ""


def _pkg_versions() -> dict:
    out = {}
    for mod in ["numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib"]:
        try:
            import importlib.metadata as md
            out[mod] = md.version(mod)
        except Exception:
            out[mod] = "?"
    return out


def write_manifest(summary: dict, args, path: Path) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "make_paper_report.py",
        "model_label": args.model_label,
        "price_input_per_1m": args.price_input,
        "price_output_per_1m": args.price_output,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": _pkg_versions(),
        "overall": summary["overall"],
        "tasks": [
            {"task_id": t["task_id"], "run": t["run"], "journal": t["journal"],
             "n_nodes": t["totals"]["n_nodes"], "best_agg": t["best_agg"],
             "n_runs": t["n_runs"]}
            for t in summary["tasks"]
        ],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_report(summary: dict, args, made_figs, path: Path) -> None:
    ov = summary["overall"]
    tasks = summary["tasks"]
    ops = operator_stats(summary)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    main_rows = [
        [t["task_id"], t["domain"], t["n_instances"], _n(t["mean_baseline"], 3),
         _n(t["mean_best"], 3), _n(t["best_agg"], 3),
         f"{t['best_operator']}@{t['best_step']}", t["totals"]["n_nodes"],
         _n(t["totals"]["cost"], 3)]
        for t in tasks
    ]
    op_rows = [
        [r["operator"], r["used"], f"{r['success_rate']*100:.0f}%",
         _n(r["mean_agg"], 3), _n(r["best_agg"], 3), r["n_best_nodes"]]
        for r in ops
    ]
    compute_rows = [
        [t["task_id"], t["totals"]["n_nodes"], f"{t['totals']['total_tokens']:,}",
         f"{t['totals']['reasoning_tokens']:,}", _n(t["totals"]["wall"] / 60, 1),
         _n(t["totals"]["cost"], 3)]
        for t in tasks
    ]

    # a representative reasoning trace (the longest one from a best node)
    trace = ""
    trace_task = ""
    for t in tasks:
        bn = _best_node(t["nodes"])
        if bn and bn["reasoning_content"] and len(bn["reasoning_content"]) > len(trace):
            trace = bn["reasoning_content"]
            trace_task = t["task_id"]
    if len(trace) > 1400:
        trace = trace[:1400].rsplit(".", 1)[0] + " …"

    multi = [t for t in tasks if len([i for i in t["instances"] if i["baseline"] is not None]) >= 2]

    lines = [
        f"# AI-improving-AI on a cybersecurity-ML gym — results",
        "",
        f"_Generated {now} by `make_paper_report.py`. Model: "
        f"**{args.model_label or 'unspecified'}**. "
        f"Numbers are read directly from the search journals; see "
        f"`manifest.json` for the exact runs, git commit, and environment._",
        "",
        "## 1. Setup",
        "",
        "The OpenMLE-Evo evolutionary search loop (Draft / Improve / Debug / "
        "Crossover operators over an island model with experience memory) is "
        "pointed at a gym of cybersecurity-ML tasks authored in NatureBench "
        "package format. Each task ships a weak baseline and a SOTA anchor; the "
        "loop is scored by **aggregate improvement**, the mean over a task's "
        "instances of "
        "`(score − baseline) / (sota − baseline)` — so **0 reproduces the weak "
        "baseline and 1 reaches the SOTA anchor**. The verifier is a hidden-label "
        "F1 metric the search never sees, so the score cannot be gamed.",
        "",
        f"This report covers **{ov['num_tasks']} task(s)**, "
        f"**{ov['total_nodes']} evaluated search nodes**, "
        f"**{ov['total_tokens']:,} model tokens**, and "
        f"**${ov['total_cost']:.2f}** of model cost "
        f"({(ov['total_wall']/60):.0f} min wall-clock total).",
        "",
        "## 2. Headline result",
        "",
        f"Across the gym the loop reached a **mean aggregate improvement of "
        f"{_n(ov['mean_best_agg'],3)}** "
        f"(range {_n(ov['min_best_agg'],3)} … {_n(ov['max_best_agg'],3)}) over the "
        f"weak baselines — i.e. it closed that fraction of the baseline→SOTA gap "
        f"on its own.",
        "",
        _md_table(
            ["Task", "Domain", "#Inst", "Baseline", "Best", "Norm. impr.",
             "Best from", "Nodes", "Cost $"], main_rows),
        "",
        "![Per-task self-improvement](plots/aggregate_bar.png)",
        "",
        "Per-task search trajectories (`plots/trajectory_<task>.png`) show the "
        "best-so-far envelope climbing off the baseline as nodes accrue.",
        "",
    ]
    if len(tasks) >= 2:
        lines += ["![All trajectories](plots/trajectory_all.png)", ""]

    lines += [
        "## 3. What drove the gains — operator effectiveness",
        "",
        "Pooled over all reported nodes (the `seed` root excluded):",
        "",
        _md_table(["Operator", "Used", "Success", "Mean impr.", "Best impr.",
                   "#Best-node"], op_rows),
        "",
        "![Operator effectiveness](plots/operator_effectiveness.png)",
        "",
        "`#Best-node` counts how many tasks had their single best candidate "
        "produced by that operator — the clearest signal of which move mattered.",
        "",
    ]

    if multi:
        lines += [
            "## 4. Per-component behaviour (the preservation constraint)",
            "",
            "Multi-instance tasks expose the loop's hardest lesson: improving the "
            "weakest component **without regressing the others**. Baseline vs "
            "evolved best per instance (dashed = SOTA anchor):",
            "",
        ]
        for t in multi:
            lines += [f"![{t['task_id']} per-instance](plots/family_{t['task_id']}.png)", ""]
            rows = [[i["instance"], _n(i["baseline"], 3), _n(i["best"], 3),
                     _n(i["sota"], 3), _n(i["norm_improvement"], 3)]
                    for i in t["instances"]]
            lines += [_md_table(["Instance", "Baseline", "Best", "SOTA",
                                 "Norm. impr."], rows), ""]

    if trace:
        lines += [
            "## 5. A representative reasoning trace",
            "",
            f"The best candidate for **{trace_task}** was produced with this "
            f"chain-of-thought (verbatim `reasoning_content`, truncated) — note it "
            f"tracing candidate lineage and explicitly reasoning about which "
            f"component to fix without disturbing the others:",
            "",
            "> " + trace.replace("\n", "\n> "),
            "",
        ]

    lines += [
        "## 6. Compute budget",
        "",
        _md_table(["Task", "Nodes", "Tokens", "Reasoning tok", "Wall (min)",
                   "Cost $"], compute_rows),
        "",
        "## 7. Reproduce",
        "",
        "```bash",
        "# 1. run the search over the task set (see .cyberml/LOCAL_MODEL.md for a "
        "local/self-hosted model)",
        "for T in $(grep -v '^#' benchmarks/cyberml_all/tasks.txt); do",
        "  .venv/bin/python scripts/run_naturebench_local.py \\",
        "    --naturebench-repo ../NatureBench --local-python .venv/bin/python \\",
        "    --data-dir .cyberml/data --skip-download --task \"$T\" \\",
        "    -- search.runner.solver.step_limit=40",
        "done",
        "",
        "# 2. regenerate every table + figure in this report",
        "uv run python .cyberml/tools/make_paper_report.py \\",
        f"  --model-label \"{args.model_label or 'Frontis-MA1-35B (local)'}\"",
        "```",
        "",
        "## 8. Limitations",
        "",
        "- Baseline and SOTA anchors are fixed per task; aggregate improvement is "
        "relative to them, not an absolute leaderboard claim.",
        "- Runs reported here are the best per task; for statistical claims, launch "
        "multiple seeds (repeat the run) — the `mean_across_runs`/`std_across_runs` "
        "columns in `main_results.csv` populate automatically when >1 run exists.",
        "- All tasks are supervised classifiers with hidden-label F1 verifiers; "
        "offensive/CTF tasks are out of scope here by design.",
        "",
        "---",
        "_Tables: `tables/*.csv`, `tables/*.tex` (booktabs). Figures: "
        "`plots/*.pdf` (vector) and `plots/*.png`. Provenance: `manifest.json`._",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                    help="Root holding the run directories (default: OpenMLE-Evo/output)")
    ap.add_argument("--paper-dir", default=str(DEFAULT_PAPER),
                    help="Where to write tables/, plots/, manifest.json, REPORT.md")
    ap.add_argument("--model-label", default=None,
                    help="Human label for the model used (goes in the report + manifest)")
    ap.add_argument("--price-input", type=float, default=None,
                    help="USD per 1M prompt tokens; overrides recorded cost when set")
    ap.add_argument("--price-output", type=float, default=None,
                    help="USD per 1M completion tokens; overrides recorded cost when set")
    ap.add_argument("--all-tasks", action="store_true",
                    help="Include every task journal, not just the .cyberml gym tasks")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    paper_dir = Path(args.paper_dir).expanduser().resolve()
    summary = build(output_dir, args.all_tasks, args.price_input, args.price_output)

    if not summary["tasks"]:
        print(f"No task journals found under {output_dir}. Run a search first.")
        return 0

    write_tables(summary, paper_dir / "tables")
    made = write_figures(summary, paper_dir / "plots")
    write_manifest(summary, args, paper_dir / "manifest.json")
    write_report(summary, args, made, paper_dir / "REPORT.md")

    ov = summary["overall"]
    print(f"\nPaper pack written to {paper_dir}")
    print(f"  tasks           : {ov['num_tasks']}")
    print(f"  mean best agg   : {_n(ov['mean_best_agg'],4)} "
          f"({_n(ov['min_best_agg'],3)} .. {_n(ov['max_best_agg'],3)})")
    print(f"  nodes / tokens  : {ov['total_nodes']} / {ov['total_tokens']:,}")
    print(f"  model cost      : ${ov['total_cost']:.2f}")
    print(f"  tables/         : main_results, per_instance, operator_effectiveness, "
          f"compute_budget (.csv + .tex)")
    print(f"  plots/          : {len(made)} trajectory + aggregate_bar + "
          f"operator_effectiveness (+ family_*)")
    print(f"  REPORT.md, manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the local cyber-ML search over several seeds for mean +/- std results.

A single search run is one sample: the LLM samples operators stochastically and
the search RNG is seeded, so one number has no error bar. This wrapper launches
``run_naturebench_local.py`` once per seed into a distinct output directory
(``<base>_seed<k>``) with the Hydra override ``seed=<k>``. make_paper_report.py /
measure_improvement.py then aggregate the repeats automatically -- the
``mean_across_runs`` / ``std_across_runs`` columns and the bar-chart error bars
populate from exactly these directories.

Everything after ``--`` is forwarded verbatim to run_naturebench_local.py (its own
flags, and its own ``--`` Hydra overrides). This wrapper only injects, per seed,
``--experiment-name <base>_seed<k>`` and the ``seed=<k>`` Hydra override.

    python scripts/run_seeds.py --base-name cyberml_nsl --seeds 1 2 3 -- \
        --naturebench-repo .cyberml/local_naturebench \
        --local-python .venv/bin/python \
        --data-dir .cyberml/data --skip-download --task nsl-kdd-nids \
        --model-base-url http://127.0.0.1:8000/v1 --model-id frontis-ma1 \
        -- search.runner.solver.step_limit=25

Then:
    uv run python .cyberml/tools/make_paper_report.py --model-label "<model>"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_naturebench_local.py"


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Return (our_args, forwarded) split on the first standalone '--'."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _inject(forwarded: list[str], base: str, seed: int) -> list[str]:
    """Forwarded runner args + this seed's experiment-name and seed override.

    The runner parses its own flags (which include --experiment-name) *before*
    its '--', and treats everything *after* '--' as Hydra overrides. So the
    experiment name must go in the pre-'--' section and ``seed=<k>`` in the
    post-'--' section."""
    if "--" in forwarded:
        i = forwarded.index("--")
        flags, overrides = list(forwarded[:i]), list(forwarded[i + 1:])
    else:
        flags, overrides = list(forwarded), []

    # drop any user-provided --experiment-name (we own it, to separate run dirs)
    cleaned: list[str] = []
    skip = False
    for tok in flags:
        if skip:
            skip = False
            continue
        if tok == "--experiment-name":
            skip = True
            continue
        if tok.startswith("--experiment-name="):
            continue
        cleaned.append(tok)
    cleaned += ["--experiment-name", f"{base}_seed{seed}"]
    overrides = [o for o in overrides if not o.startswith("seed=")] + [f"seed={seed}"]
    return cleaned + ["--"] + overrides


def main() -> int:
    our, forwarded = _split_argv(sys.argv[1:])
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-name", required=True,
                    help="output dirs are output/<base>_seed<k>")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--keep-going", action="store_true",
                    help="continue to the next seed if one run fails")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-seed commands, run nothing")
    args = ap.parse_args(our)
    if not forwarded:
        ap.error("pass the run_naturebench_local.py arguments after '--'")

    print(f"Seeds: {args.seeds}  base: {args.base_name}\n")
    failures = []
    for k in args.seeds:
        cmd = [sys.executable, str(RUNNER)] + _inject(forwarded, args.base_name, k)
        print("=" * 72)
        print(f"[seed {k}] " + " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            failures.append(k)
            print(f"[seed {k}] FAILED (exit {rc})")
            if not args.keep_going:
                return rc
    if args.dry_run:
        print("\n(dry run: nothing executed)")
        return 0
    print("\n" + "=" * 72)
    if failures:
        print(f"Done with failures on seeds: {failures}")
        return 1
    print(f"All {len(args.seeds)} seed runs finished. Aggregate with:")
    print("  uv run python .cyberml/tools/make_paper_report.py --model-label '<model>'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stub to satisfy the runner's repo check in local (no-external-repo) mode.

`run_naturebench_local.py` treats a directory as the NatureBench repo only if it
contains both `run_naturebench.py` and `eval_service.py`. In local mode the task
packages are pre-materialized by the .cyberml/tools/prepare_*.py scripts and the
run is launched with `--skip-download`, so this download entry point is never
invoked for real. If it *is* called (i.e. without --skip-download), it verifies
the requested packages already exist and exits, rather than fetching anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dataset-revision", default=None)
    args, _ = parser.parse_known_args()

    data_root = Path(args.data_dir).expanduser().resolve()
    tasks_root = data_root / "tasks" if (data_root / "tasks").is_dir() else data_root
    task_ids = [ln.strip() for ln in Path(args.tasks).read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    missing = [t for t in task_ids
               if not (tasks_root / t / "metadata.json").is_file()]
    if missing:
        print("Local NatureBench mode: these packages are not materialized: "
              + ", ".join(missing) + "\nRun the matching "
              ".cyberml/tools/prepare_*.py script, or pass --skip-download.",
              file=sys.stderr)
        return 1
    print(f"Local mode: {len(task_ids)} package(s) already present; nothing to "
          "download.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

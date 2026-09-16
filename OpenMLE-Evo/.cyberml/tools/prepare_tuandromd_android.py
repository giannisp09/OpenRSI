#!/usr/bin/env python3
"""Materialize the TUANDROMD Android-malware detection task package.

Fourth cyber-ML task, a *different domain and feature space* again: static
Android app analysis via permission + API-call presence flags (no network flows,
no PE headers, no URL strings). Single binary instance `main`: predict malware
(`1`) vs. goodware (`0`) from 241 binary features. Same interface as the other
tasks (predictions.csv with a `label_pred` column, metric = Detection F1-Score).

Dataset: TUANDROMD (UCI ML Repository, id 855) — 4,464 Android apps, 241 binary
permission / decompiled-API-call features, imbalanced toward malware (~80%).

Output layout:
    <task>/problem/data/main/{train.csv, x_test.csv}
    <task>/evaluation/ground_truth/main/y_ref.csv
    <task>/metadata.json
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
TASK_DIR = TOOLS_DIR.parent / "data" / "tasks" / "tuandromd-android"
CACHE_DIR = TASK_DIR.parent / ".tuandromd_cache"
SOURCE = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00622/"
          "TUANDROMD.csv")

LABEL_COL = "Label"             # strings: "malware" / "goodware"
LABEL_MAP = {"malware": 1, "goodware": 0}
CATEGORICAL: list[str] = []    # all 241 features are binary 0/1
SEED = 42
TEST_FRAC = 0.2


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "openrsi-cyberml"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())


def _baseline_f1(train, x_test, y_test, features):
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=200)),
    ])
    clf.fit(train[features], train["label"])
    pred = clf.predict(x_test[features])
    return float(f1_score(y_test, pred, pos_label=1, zero_division=0))


def main() -> int:
    print("Loading TUANDROMD dataset ...")
    raw = CACHE_DIR / "tuandromd.csv"
    _download(SOURCE, raw)
    df = pd.read_csv(raw)
    # Exactly one row has an empty/NaN Label in the source; drop before mapping.
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    df = df[df[LABEL_COL].astype(str).str.strip() != ""].reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].map(LABEL_MAP).astype(int)
    features = [c for c in df.columns if c != LABEL_COL]

    train, test = train_test_split(df, test_size=TEST_FRAC, random_state=SEED,
                                   stratify=df[LABEL_COL])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    data_root = TASK_DIR / "problem" / "data" / "main"
    gt_root = TASK_DIR / "evaluation" / "ground_truth" / "main"
    data_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    train[features + [LABEL_COL]].rename(columns={LABEL_COL: "label"}).to_csv(
        data_root / "train.csv", index=False)
    test[features].to_csv(data_root / "x_test.csv", index=False)
    test[[LABEL_COL]].rename(columns={LABEL_COL: "label"}).to_csv(
        gt_root / "y_ref.csv", index=False)

    base = _baseline_f1(train.rename(columns={LABEL_COL: "label"}),
                        test, test[LABEL_COL], features)
    pos = int(test[LABEL_COL].sum())
    print(f"  [main ] train={len(train):6d} test={len(test):6d} "
          f"malware_in_test={pos:6d} features={len(features)} "
          f"baseline_f1={base:.4f}")
    sota = round(min(0.999, max(base + 0.03, 0.99)), 4)

    metadata = {
        "task_name": "TUANDROMD Android Malware Detection",
        "workflow_topology": "strict_single_step",
        "methodology_paradigm": "general_ml_application",
        "tooling_metadata": None,
        "domain_metadata": {
            "primary_domain": "Cybersecurity",
            "sub_domain": "Malware Detection",
            "domain_tags": ["android malware", "static analysis", "permissions",
                            "API calls", "binary classification", "TUANDROMD"],
        },
        "compute_resource_requirements": {
            "cpu_compute": {"severity": "low",
                            "quantity_text": "Standard personal machine; runs on CPU"},
            "gpu_compute": {"severity": "low",
                            "quantity_text": "Optional; Apple MPS / CUDA usable for DL models"},
            "runtime": {"severity": "short",
                        "quantity_text": "~1 minute per candidate with sklearn models"},
        },
        "performance_entries": [{
            "dataset_name": "main",
            "metrics": [{
                "name": "Detection F1-Score",
                "is_primary": True,
                "metric_direction": "higher_is_better",
                "source_description": "Binary goodware-vs-malware F1 on a stratified 20% test split",
                "unit": None,
                "sota_score": [{"value": f"{sota:.4f}",
                                "method": "Strong tree ensemble (illustrative anchor)"}],
                "baseline_score": {"value": f"{base:.4f}",
                                   "method": "Weak logistic-regression baseline"},
            }],
        }],
    }
    (TASK_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nWrote {TASK_DIR / 'metadata.json'}")
    print("Done. Task package materialized at", TASK_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

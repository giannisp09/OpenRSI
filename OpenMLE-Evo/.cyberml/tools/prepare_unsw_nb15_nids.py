#!/usr/bin/env python3
"""Materialize the UNSW-NB15 network-intrusion task package.

Fourth cyber-ML task: modern network-flow intrusion detection. A *different
feature space* from NSL-KDD (richer flow/connection statistics, newer attack
mix), so it broadens the network-IDS coverage and the transfer curriculum. Same
interface as the other tasks (binary detection -> predictions.csv with a
`label_pred` column, metric = Detection F1-Score), single instance `main`.

Dataset: UNSW-NB15 (Moustafa & Slay, Australian Centre for Cyber Security). We
use the testing-set CSV (~82k flows, 45 columns). Two columns are dropped:
`id` (a row index) and `attack_cat` (the multiclass attack category, which would
leak the binary `label`). The remaining 42 columns are the features; `label`
(0 = normal, 1 = attack) is the target.

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

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

TOOLS_DIR = Path(__file__).resolve().parent
TASK_DIR = TOOLS_DIR.parent / "data" / "tasks" / "unsw-nb15-nids"
CACHE_DIR = TASK_DIR.parent / ".unsw_cache"
SOURCE = ("https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
          "UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv")

LABEL_COL = "label"                       # 1 = attack, 0 = normal
DROP_COLS = ["id", "attack_cat"]          # index + multiclass leak of label
CATEGORICAL = ["proto", "service", "state"]
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
    numeric = [c for c in features if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=-1))]), CATEGORICAL),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric),
    ])
    clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=200))])
    clf.fit(train[features], train["label"])
    pred = clf.predict(x_test[features])
    return float(f1_score(y_test, pred, pos_label=1, zero_division=0))


def main() -> int:
    print("Loading UNSW-NB15 (testing-set) ...")
    raw = CACHE_DIR / "unsw_nb15_testing.csv"
    _download(SOURCE, raw)
    df = pd.read_csv(raw, encoding="utf-8-sig")      # file carries a UTF-8 BOM
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype(str)
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
          f"attack_in_test={pos:6d} features={len(features)} "
          f"baseline_f1={base:.4f}")
    sota = round(min(0.999, max(base + 0.03, 0.99)), 4)

    metadata = {
        "task_name": "UNSW-NB15 Network Intrusion Detection",
        "workflow_topology": "strict_single_step",
        "methodology_paradigm": "general_ml_application",
        "tooling_metadata": None,
        "domain_metadata": {
            "primary_domain": "Cybersecurity",
            "sub_domain": "Network Intrusion Detection",
            "domain_tags": ["network intrusion detection", "flow features",
                            "UNSW-NB15", "binary classification"],
        },
        "compute_resource_requirements": {
            "cpu_compute": {"severity": "low",
                            "quantity_text": "Standard personal machine; runs on CPU"},
            "gpu_compute": {"severity": "low",
                            "quantity_text": "Optional; Apple MPS / CUDA usable for DL models"},
            "runtime": {"severity": "short",
                        "quantity_text": "~1-2 minutes per candidate with sklearn models"},
        },
        "performance_entries": [{
            "dataset_name": "main",
            "metrics": [{
                "name": "Detection F1-Score",
                "is_primary": True,
                "metric_direction": "higher_is_better",
                "source_description": "Binary normal-vs-attack F1 on a stratified 20% test split",
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

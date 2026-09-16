#!/usr/bin/env python3
"""Materialize the DGA-domain detection task package.

A cyber-ML task with a *deliberately raw feature space*: the only input is the
domain **string** itself. The candidate must engineer features from it (length,
character entropy, n-gram statistics, vowel/digit ratios, dictionary-likeness,
...). This exercises the search on feature engineering, not just model choice —
a different lever from the other tasks, whose features are pre-computed.

Single binary instance `main`: predict whether a domain name was produced by a
malware **Domain Generation Algorithm** (`1` = dga) or is a benign/real domain
(`0` = legit). Same interface as the other tasks (predictions.csv with a
`label_pred` column, metric = Detection F1-Score).

Dataset: chrmor/DGA_domains_dataset (sample) — 9,999 domains, balanced, benign
class from Alexa top sites, DGA class from 25+ malware families (Netlab 360).
The raw CSV is headerless: columns are (label, family, domain). We keep only the
domain string and the binary label; `family` is dropped (it leaks the label).

Output layout:
    <task>/problem/data/main/{train.csv, x_test.csv}   # columns: domain[,label]
    <task>/evaluation/ground_truth/main/y_ref.csv
    <task>/metadata.json
"""
from __future__ import annotations

import json
import math
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TOOLS_DIR = Path(__file__).resolve().parent
TASK_DIR = TOOLS_DIR.parent / "data" / "tasks" / "dga-domains-dns"
CACHE_DIR = TASK_DIR.parent / ".dga_cache"
SOURCE = ("https://raw.githubusercontent.com/chrmor/DGA_domains_dataset/"
          "master/dga_domains_sample.csv")

SEED = 42
TEST_FRAC = 0.2


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "openrsi-cyberml"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _simple_features(domains: pd.Series) -> pd.DataFrame:
    """The weak baseline's hand-features (candidates should do better)."""
    d = domains.astype(str).str.lower()
    vowels = set("aeiou")
    return pd.DataFrame({
        "length": d.str.len(),
        "num_digits": d.str.count(r"\d"),
        "digit_ratio": d.str.count(r"\d") / d.str.len().clip(lower=1),
        "num_dots": d.str.count(r"\."),
        "entropy": d.map(_shannon_entropy),
        "vowel_ratio": d.map(lambda s: sum(c in vowels for c in s) / max(len(s), 1)),
        "distinct_ratio": d.map(lambda s: len(set(s)) / max(len(s), 1)),
    })


def main() -> int:
    print("Loading DGA-domains dataset ...")
    raw = CACHE_DIR / "dga_domains_sample.csv"
    _download(SOURCE, raw)
    df = pd.read_csv(raw, header=None, names=["label", "family", "domain"])
    df = df.dropna(subset=["label", "domain"]).reset_index(drop=True)
    df["label"] = (df["label"].astype(str).str.strip().str.lower() == "dga").astype(int)
    df["domain"] = df["domain"].astype(str)
    df = df[["domain", "label"]]

    train, test = train_test_split(df, test_size=TEST_FRAC, random_state=SEED,
                                   stratify=df["label"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    data_root = TASK_DIR / "problem" / "data" / "main"
    gt_root = TASK_DIR / "evaluation" / "ground_truth" / "main"
    data_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    train[["domain", "label"]].to_csv(data_root / "train.csv", index=False)
    test[["domain"]].to_csv(data_root / "x_test.csv", index=False)
    test[["label"]].to_csv(gt_root / "y_ref.csv", index=False)

    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=200))])
    clf.fit(_simple_features(train["domain"]), train["label"])
    pred = clf.predict(_simple_features(test["domain"]))
    base = float(f1_score(test["label"], pred, pos_label=1, zero_division=0))
    pos = int(test["label"].sum())
    print(f"  [main ] train={len(train):5d} test={len(test):5d} "
          f"dga_in_test={pos:5d} baseline_f1={base:.4f}")
    sota = round(min(0.999, max(base + 0.08, 0.97)), 4)

    metadata = {
        "task_name": "DGA Domain Detection",
        "workflow_topology": "strict_single_step",
        "methodology_paradigm": "general_ml_application",
        "tooling_metadata": None,
        "domain_metadata": {
            "primary_domain": "Cybersecurity",
            "sub_domain": "DNS / Malware C2 Detection",
            "domain_tags": ["DGA detection", "DNS security", "domain generation",
                            "string features", "feature engineering",
                            "binary classification"],
        },
        "compute_resource_requirements": {
            "cpu_compute": {"severity": "low",
                            "quantity_text": "Standard personal machine; runs on CPU"},
            "gpu_compute": {"severity": "low",
                            "quantity_text": "Optional; a char-level model may use MPS/CUDA"},
            "runtime": {"severity": "short",
                        "quantity_text": "~1 minute per candidate with engineered features"},
        },
        "performance_entries": [{
            "dataset_name": "main",
            "metrics": [{
                "name": "Detection F1-Score",
                "is_primary": True,
                "metric_direction": "higher_is_better",
                "source_description": "Binary DGA-vs-legit F1 on a stratified 20% test split",
                "unit": None,
                "sota_score": [{"value": f"{sota:.4f}",
                                "method": "Char-level features / n-gram model (illustrative anchor)"}],
                "baseline_score": {"value": f"{base:.4f}",
                                   "method": "Weak LR on 7 hand-built lexical features"},
            }],
        }],
    }
    (TASK_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nWrote {TASK_DIR / 'metadata.json'}")
    print("Done. Task package materialized at", TASK_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

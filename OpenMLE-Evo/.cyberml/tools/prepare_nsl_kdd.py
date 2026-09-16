#!/usr/bin/env python3
"""Materialize the NSL-KDD network-intrusion-detection task package.

Downloads NSL-KDD (KDDTrain+ / KDDTest+), maps the fine-grained attack labels
onto the four standard families (DoS, Probe, R2L, U2R), and writes one *instance*
per family as an independent binary detection subproblem (normal vs. that family).

Output layout (mirrors an OpenMLE-Evo / NatureBench task package):

    <task>/problem/data/<instance>/train.csv     # features + binary `label`
    <task>/problem/data/<instance>/x_test.csv     # features only (no label)
    <task>/evaluation/ground_truth/<instance>/y_ref.csv   # hidden `label`
    <task>/metadata.json                          # metric + per-instance anchors

The four families deliberately span the difficulty spectrum: DoS/Probe are easy
and saturate quickly, R2L/U2R are hard and carry the remaining headroom. That
reproduces the per-component / preservation dynamic documented in
PORTING-OPENMLE.md section 4, which is the point of using four instances rather
than one aggregate binary task.

Idempotent: raw files are cached under <task>/../.nsl_kdd_cache/. Re-running
rebuilds the splits deterministically (fixed seed).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# --- Paths -------------------------------------------------------------------
TOOLS_DIR = Path(__file__).resolve().parent
TASK_DIR = TOOLS_DIR.parent / "data" / "tasks" / "nsl-kdd-nids"
CACHE_DIR = TASK_DIR.parent / ".nsl_kdd_cache"

# --- Data source (public mirror of the UNB NSL-KDD distribution) -------------
MIRROR = "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master"
SOURCES = {
    "train": f"{MIRROR}/NSL_KDD_Train.csv",
    "test": f"{MIRROR}/NSL_KDD_Test.csv",
}

# --- Schema ------------------------------------------------------------------
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
COLUMNS = FEATURE_NAMES + ["attack"]  # the mirror CSV has 41 features + attack name

# --- Standard NSL-KDD family mapping -----------------------------------------
FAMILIES = {
    "dos": {"apache2", "back", "land", "neptune", "mailbomb", "pod",
            "processtable", "smurf", "teardrop", "udpstorm", "worm"},
    "probe": {"ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"},
    "r2l": {"ftp_write", "guess_passwd", "httptunnel", "imap", "multihop",
            "named", "phf", "sendmail", "snmpgetattack", "snmpguess", "spy",
            "warezclient", "warezmaster", "xlock", "xsnoop"},
    "u2r": {"buffer_overflow", "loadmodule", "perl", "ps", "rootkit",
            "sqlattack", "xterm"},
}
INSTANCES = ["dos", "probe", "r2l", "u2r"]
SEED = 42
# Normal rows are capped relative to the family's attack count so that rare
# families (U2R, R2L) stay learnable instead of drowning in 67k normals.
NORMAL_RATIO = 5
NORMAL_MIN, NORMAL_MAX = 2000, 20000


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "openrsi-cyberml"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def _load(split: str) -> pd.DataFrame:
    dest = CACHE_DIR / f"nsl_kdd_{split}.csv"
    _download(SOURCES[split], dest)
    df = pd.read_csv(dest, header=None)
    # The mirror sometimes ships 42 cols (41 feat + attack) and sometimes 43
    # (+ difficulty). Keep the first 42 either way.
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    df["attack"] = df["attack"].astype(str).str.strip().str.rstrip(".")
    return df


def _family_of(attack: str) -> str | None:
    if attack == "normal":
        return "normal"
    for fam, members in FAMILIES.items():
        if attack in members:
            return fam
    return None  # unmapped labels are dropped (keeps the split well-defined)


def _instance_frame(df: pd.DataFrame, family: str, rng: np.random.Generator,
                    normal_pool: pd.DataFrame) -> pd.DataFrame:
    fam_rows = df[df["attack"].map(_family_of) == family].copy()
    n_norm = int(np.clip(NORMAL_RATIO * len(fam_rows), NORMAL_MIN, NORMAL_MAX))
    n_norm = min(n_norm, len(normal_pool))
    norm_rows = normal_pool.sample(n=n_norm, random_state=int(rng.integers(1e9)))
    fam_rows["label"] = 1
    norm_rows = norm_rows.copy()
    norm_rows["label"] = 0
    out = pd.concat([fam_rows, norm_rows], ignore_index=True)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


def _baseline_f1(train: pd.DataFrame, test_x: pd.DataFrame,
                 test_y: pd.Series) -> float:
    """A deliberately weak logistic-regression baseline the search should beat."""
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), CATEGORICAL),
         ("num", StandardScaler(), [c for c in FEATURE_NAMES
                                    if c not in CATEGORICAL])])
    clf = Pipeline([("pre", pre),
                    ("lr", LogisticRegression(max_iter=200, n_jobs=None))])
    clf.fit(train[FEATURE_NAMES], train["label"])
    pred = clf.predict(test_x[FEATURE_NAMES])
    return float(f1_score(test_y, pred, pos_label=1, zero_division=0))


def main() -> int:
    print("Loading NSL-KDD ...")
    train_all = _load("train")
    test_all = _load("test")
    train_norm = train_all[train_all["attack"] == "normal"]
    test_norm = test_all[test_all["attack"] == "normal"]
    rng = np.random.default_rng(SEED)

    data_root = TASK_DIR / "problem" / "data"
    gt_root = TASK_DIR / "evaluation" / "ground_truth"

    perf_entries = []
    for inst in INSTANCES:
        tr = _instance_frame(train_all, inst, rng, train_norm)
        te = _instance_frame(test_all, inst, rng, test_norm)

        # visible train (features + label), visible x_test (features only),
        # hidden y_ref (label).
        (data_root / inst).mkdir(parents=True, exist_ok=True)
        (gt_root / inst).mkdir(parents=True, exist_ok=True)
        tr[FEATURE_NAMES + ["label"]].to_csv(data_root / inst / "train.csv",
                                             index=False)
        te[FEATURE_NAMES].to_csv(data_root / inst / "x_test.csv", index=False)
        te[["label"]].to_csv(gt_root / inst / "y_ref.csv", index=False)

        base = _baseline_f1(tr, te, te["label"])
        pos = int(te["label"].sum())
        print(f"  [{inst:5}] train={len(tr):6d} test={len(te):6d} "
              f"attacks_in_test={pos:5d} baseline_f1={base:.4f}")

        # SOTA anchor: a strong-but-reachable target above the weak baseline, so
        # aggregate_improvement has real gradient (see PORTING-OPENMLE.md sec 4).
        sota = round(min(0.995, max(base + 0.15, 0.90)), 4)
        perf_entries.append({
            "dataset_name": inst,
            "metrics": [{
                "name": "Detection F1-Score",
                "is_primary": True,
                "metric_direction": "higher_is_better",
                "source_description": "Binary normal-vs-attack F1 on KDDTest+ subset",
                "unit": None,
                "sota_score": [{"value": f"{sota:.4f}",
                                "method": "Strong tree ensemble (illustrative anchor)"}],
                "baseline_score": {"value": f"{base:.4f}",
                                   "method": "Weak logistic-regression baseline"},
            }],
        })

    metadata = {
        "task_name": "NSL-KDD Network Intrusion Detection (per-family)",
        "workflow_topology": "strict_single_step",
        "methodology_paradigm": "general_ml_application",
        "tooling_metadata": None,
        "domain_metadata": {
            "primary_domain": "Cybersecurity",
            "sub_domain": "Network Intrusion Detection",
            "domain_tags": ["intrusion detection", "network security", "NSL-KDD",
                            "anomaly detection", "supervised classification"],
        },
        "compute_resource_requirements": {
            "cpu_compute": {"severity": "low",
                            "quantity_text": "Standard personal machine; runs on CPU"},
            "gpu_compute": {"severity": "low",
                            "quantity_text": "Optional; Apple MPS / CUDA usable for DL models"},
            "runtime": {"severity": "short",
                        "quantity_text": "~1-3 minutes per candidate with sklearn models"},
        },
        "performance_entries": perf_entries,
    }
    (TASK_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nWrote {TASK_DIR / 'metadata.json'}")
    print("Done. Task package materialized at", TASK_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

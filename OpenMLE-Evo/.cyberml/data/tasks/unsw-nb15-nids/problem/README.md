# UNSW-NB15 Network Intrusion Detection

## 1. Task

Detect malicious network flows in the UNSW-NB15 dataset. One binary instance,
`main`: given 42 flow/connection features for a network flow, predict **attack
(`1`) vs. normal (`0`)**. Score is the F1-score on the attack class.

This is a network-intrusion task like `nsl-kdd-nids`, but on a **newer, richer
feature space** — modern flow statistics and connection-count features, with a
different attack mix. The interface is identical, so it extends the transfer
curriculum and tests whether operator experience carries across two different
network-IDS feature spaces. A weak logistic-regression baseline is already
reasonably strong; the task rewards better categorical handling, stronger models,
and threshold calibration.

## 2. Data

Read inputs from the `DATA_DIR` environment variable:

- `DATA_DIR/main/train.csv` — 42 feature columns + a binary `label` column
  (`1` = attack, `0` = normal).
- `DATA_DIR/main/x_test.csv` — the same 42 feature columns, no label.

Three columns are categorical strings — `proto` (transport protocol, e.g. `tcp`,
`udp`), `service` (application service, e.g. `http`, `dns`, `-`), and `state`
(connection state, e.g. `FIN`, `CON`, `INT`). The remaining 39 features are
numeric flow/timing/count statistics. The test set may contain `service` /
`state` / `proto` values unseen in training — **encode robustly**. The original
`id` (row index) and `attack_cat` (multiclass attack category) columns are
removed; `attack_cat` is dropped specifically because it would leak the binary
label. See `data_description.md` for the full schema.

## 3. Output format (strict)

Write exactly:

    OUTPUT_DIR/main/predictions.csv

with a single integer column **`label_pred`** (`0` or `1`), **one row per row of
`x_test.csv`, in the same order**. `OUTPUT_DIR` is an environment variable.

## 4. Metric

Primary metric: **`Detection F1-Score`** = binary F1 with `pos_label=1`
(attack), higher is better. Improvement is measured against the weak-LR baseline
in `metadata.json`.

## 5. Notes on compute

Runs on CPU in a minute or two (~82k flows × 42 features). Optional
acceleration: Apple **MPS** (`torch.backends.mps.is_available()`) or CUDA for
deep models — but gradient-boosted trees / random forests are strong and fast on
this tabular set. The unseen-category caveat matters: encode `proto` / `service`
/ `state` so unseen test values do not crash or silently corrupt the mapping.

## 6. Entry point (`run.py`)

A complete, runnable baseline follows. It handles the three categorical columns
robustly (unseen values map to a reserved code). Improve via stronger models,
class balancing, feature engineering, and threshold tuning; keep the Section 3
output contract exact.

```python
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

DATA_DIR = os.environ["DATA_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
CATEGORICAL = ["proto", "service", "state"]


def build_model(feature_names):
    numeric = [c for c in feature_names if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CATEGORICAL),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=200,
                                                class_weight="balanced"))])


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "main", "train.csv"))
    x_test = pd.read_csv(os.path.join(DATA_DIR, "main", "x_test.csv"))
    feature_names = [c for c in train.columns if c != "label"]

    model = build_model(feature_names)
    model.fit(train[feature_names], train["label"])
    preds = model.predict(x_test[feature_names]).astype(int)

    out_dir = os.path.join(OUTPUT_DIR, "main")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"label_pred": preds}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False)
    print(f"[main] wrote {len(preds)} predictions")


if __name__ == "__main__":
    main()
```

# NSL-KDD Network Intrusion Detection (per-family)

## 1. Task

Build a network intrusion detector. The task is split into **four independent
binary sub-problems ("instances"), one per NSL-KDD attack family**:

| Instance | Family | Difficulty |
| --- | --- | --- |
| `dos`   | Denial of Service (neptune, smurf, back, teardrop, …) | easy |
| `probe` | Probing / scanning (satan, ipsweep, portsweep, nmap, …) | easy |
| `r2l`   | Remote-to-Local (guess_passwd, warezclient, …) | hard |
| `u2r`   | User-to-Root (buffer_overflow, rootkit, loadmodule, …) | hard, few samples |

For each instance you train on `train.csv` (normal traffic vs. that attack
family) and predict on `x_test.csv`. The score is the **F1-score on the attack
class (label = 1)**, averaged across the four instances
(`aggregate_improvement`).

`dos` and `probe` are easy and will saturate early; `r2l` and `u2r` are hard and
carry most of the remaining headroom. **Improving the hard instances must not
regress the easy ones** — because the reported score is a mean, a regression on a
solved instance costs more than the available gain on a weak one. Preserve what
already works while you push the weak family.

## 2. Data

Read all inputs from the directory named by the `DATA_DIR` environment variable.
Per instance `<inst>` in `{dos, probe, r2l, u2r}`:

- `DATA_DIR/<inst>/train.csv` — 41 feature columns + a binary `label` column
  (`1` = attack of this family, `0` = normal).
- `DATA_DIR/<inst>/x_test.csv` — the same 41 feature columns, **no label**.

Three columns are categorical strings (`protocol_type`, `service`, `flag`); the
rest are numeric. See `data_description.md` for the full schema. The test set may
contain `service`/`flag` values unseen in training — encode robustly.

## 3. Output format (strict)

For each instance, write exactly:

    OUTPUT_DIR/<inst>/predictions.csv

with a single integer column **`label_pred`** (values `0` or `1`), **one row per
row of `x_test.csv`, in the same order**. `OUTPUT_DIR` is provided as an
environment variable. Create the per-instance subdirectories. Nothing else in the
output directory is read.

## 4. Metric

Primary metric per instance: **`Detection F1-Score`** = binary F1 with
`pos_label=1`, higher is better. The search maximizes the mean improvement over
the per-instance baselines (a weak logistic-regression baseline; see
`metadata.json`).

## 5. Notes on compute

Runs on CPU. Optional acceleration is available: this machine exposes Apple
**MPS** (`torch.backends.mps.is_available()`), and CUDA where present — a deep
model may use `torch.device("mps")`. Classical models (gradient-boosted trees,
random forests, calibrated linear models) are strong on NSL-KDD and fast on CPU;
prefer them unless a deep model clearly helps a hard family.

## 6. Entry point (`run.py`)

A complete, runnable baseline follows. It is intentionally weak (one shared
logistic-regression pipeline per instance) so there is headroom to improve —
better feature handling, class weighting/resampling for `r2l`/`u2r`, and stronger
models are all open. Keep the output contract in Section 3 exact.

```python
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

DATA_DIR = os.environ["DATA_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
INSTANCES = ["dos", "probe", "r2l", "u2r"]

CATEGORICAL = ["protocol_type", "service", "flag"]


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
    for inst in INSTANCES:
        train = pd.read_csv(os.path.join(DATA_DIR, inst, "train.csv"))
        x_test = pd.read_csv(os.path.join(DATA_DIR, inst, "x_test.csv"))
        feature_names = [c for c in train.columns if c != "label"]

        model = build_model(feature_names)
        model.fit(train[feature_names], train["label"])
        preds = model.predict(x_test[feature_names]).astype(int)

        out_dir = os.path.join(OUTPUT_DIR, inst)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"label_pred": preds}).to_csv(
            os.path.join(out_dir, "predictions.csv"), index=False)
        print(f"[{inst}] wrote {len(preds)} predictions")


if __name__ == "__main__":
    main()
```

# TUANDROMD Android Malware Detection

## 1. Task

Detect Android malware from static app features. One binary instance, `main`:
given 241 binary presence flags (requested permissions + decompiled API-call
signatures) for an Android app, predict **malware (`1`) vs. goodware (`0`)**.
Score is the F1-score on the malware class.

This is a fourth cyber-ML task in yet another **domain and feature space** —
neither network flows (`nsl-kdd-nids`), PE headers (`clamp-pe-malware`), nor URL
strings (`phishing-url`) but static Android app manifests / bytecode. The
interface is identical, so it extends the transfer curriculum. The feature space
is **high-dimensional and sparse** (mostly 0/1 flags) and the dataset is
**imbalanced toward malware (~80%)**; the task rewards regularization, feature
selection, and class-aware thresholding rather than a dramatic climb.

## 2. Data

Read inputs from the `DATA_DIR` environment variable:

- `DATA_DIR/main/train.csv` — 241 feature columns + a binary `label` column
  (`1` = malware, `0` = goodware).
- `DATA_DIR/main/x_test.csv` — the same 241 feature columns, no label.

All 241 features are binary (`0`/`1`) presence flags: Android permissions
(e.g. `SEND_SMS`, `ACCESS_FINE_LOCATION`, `READ_CONTACTS`, `INSTALL_PACKAGES`)
and decompiled API-call signatures (e.g. `Landroid/...` or
`Lorg/apache/http/...->method`). A `1` means the app requests the permission /
invokes the API; `0` means it does not. See `data_description.md`.

## 3. Output format (strict)

Write exactly:

    OUTPUT_DIR/main/predictions.csv

with a single integer column **`label_pred`** (`0` or `1`), **one row per row of
`x_test.csv`, in the same order**. `OUTPUT_DIR` is an environment variable.

## 4. Metric

Primary metric: **`Detection F1-Score`** = binary F1 with `pos_label=1`
(malware), higher is better. Improvement is measured against the weak-LR baseline
in `metadata.json`.

## 5. Notes on compute

Runs on CPU in well under a minute (~4.5k rows × 241 binary features). Optional
acceleration: Apple **MPS** (`torch.backends.mps.is_available()`) or CUDA for
deep models — but on this small sparse tabular set, regularized linear models and
gradient-boosted trees are strong and fast. Because features are already `0`/`1`,
scaling matters little for trees; L1/L2 regularization or feature selection helps
most in this high-dimensional sparse space, and the ~80/20 class imbalance makes
`class_weight`/threshold choices matter.

## 6. Entry point (`run.py`)

A complete, runnable baseline follows. It is intentionally plain (a scaled
logistic regression) so there is headroom. Improve via stronger models, feature
selection, regularization, and threshold tuning; keep the Section 3 output
contract exact.

```python
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.environ["DATA_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]


def build_model():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
    ])


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "main", "train.csv"))
    x_test = pd.read_csv(os.path.join(DATA_DIR, "main", "x_test.csv"))
    feature_names = [c for c in train.columns if c != "label"]

    model = build_model()
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

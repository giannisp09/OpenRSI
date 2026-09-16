# Phishing-URL Detection

## 1. Task

Detect phishing URLs from lexical and host-based features. One binary instance,
`main`: given 111 numeric features derived from a URL (and its domain / DNS /
traffic lookups), predict **phishing (`1`) vs. legitimate (`0`)**. Score is the
F1-score on the phishing class.

This is a third cyber-ML task in yet another **domain and feature space** —
neither network flows (`nsl-kdd-nids`) nor PE headers (`clamp-pe-malware`) but URL
strings and their web/DNS context. The interface is identical, so it extends the
transfer curriculum. The dataset is balanced (~52% phishing) and a weak
logistic-regression baseline is already fairly strong; the task rewards better
models, feature interactions, and threshold calibration.

## 2. Data

Read inputs from the `DATA_DIR` environment variable:

- `DATA_DIR/main/train.csv` — 111 feature columns + a binary `label` column
  (`1` = phishing, `0` = legitimate).
- `DATA_DIR/main/x_test.csv` — the same 111 feature columns, no label.

All 111 features are numeric. They are counts of characters in URL components
(`qty_dot_url`, `qty_hyphen_url`, `qty_slash_url`, …, and the same per
`_domain` / `_directory` / `_file` / `_params`), lengths (`length_url`,
`domain_length`), boolean flags (`domain_in_ip`, `email_in_url`,
`url_shortened`, `tls_ssl_certificate`), and host/DNS/traffic signals
(`time_domain_activation`, `qty_nameservers`, `qty_mx_servers`, `ttl_hostname`,
`domain_spf`, `asn_ip`, `qty_redirects`, `url_google_index`,
`domain_google_index`). A value of `-1` marks a feature that could not be
resolved (e.g. a DNS/WHOIS lookup that failed); treat `-1` as "unknown", not as a
magnitude. See `data_description.md`.

## 3. Output format (strict)

Write exactly:

    OUTPUT_DIR/main/predictions.csv

with a single integer column **`label_pred`** (`0` or `1`), **one row per row of
`x_test.csv`, in the same order**. `OUTPUT_DIR` is an environment variable.

## 4. Metric

Primary metric: **`Detection F1-Score`** = binary F1 with `pos_label=1`
(phishing), higher is better. Improvement is measured against the weak-LR baseline
in `metadata.json`.

## 5. Notes on compute

Runs on CPU in a minute or two (58k rows × 111 features). Optional acceleration:
Apple **MPS** (`torch.backends.mps.is_available()`) or CUDA for deep models — but
gradient-boosted trees / random forests are strong and fast on this tabular set.
The `-1`-as-unknown convention matters: consider missing-value indicators or
per-feature handling rather than feeding `-1` as a real number to a linear model.

## 6. Entry point (`run.py`)

A complete, runnable baseline follows. It is intentionally plain (a scaled
logistic regression) so there is headroom. Improve via stronger models,
`-1`-aware preprocessing, feature engineering, and threshold tuning; keep the
Section 3 output contract exact.

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
        ("clf", LogisticRegression(max_iter=200)),
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

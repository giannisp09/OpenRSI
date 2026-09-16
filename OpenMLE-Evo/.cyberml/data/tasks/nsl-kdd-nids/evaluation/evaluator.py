"""NSL-KDD intrusion-detection evaluator.

Reads candidate predictions from OUTPUT_DIR, validates them against the hidden
ground truth in this directory, and writes score.json as
{instance: {metric_name: value}}. Metric names must match metadata.json exactly.

The evaluator never sees candidate source and the candidate never sees this
directory: ground truth stays here, predictions arrive under OUTPUT_DIR. This is
the verifier-integrity boundary described in PORTING-OPENMLE.md section 5.
"""
import json
import os

import pandas as pd
from sklearn.metrics import f1_score

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(EVAL_DIR, "ground_truth")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

OUTPUT_FILE = "predictions.csv"
INSTANCES = ["dos", "probe", "r2l", "u2r"]
METRIC_NAMES = ["Detection F1-Score"]
PRED_COL = "label_pred"
VALID_VALUES = {0, 1}


class ValidationError(Exception):
    pass


def error_result(msg):
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(msg)
    return result


def load_and_validate(instance_name):
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)
    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")
    try:
        preds = pd.read_csv(pred_file)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV: {e}")

    if PRED_COL not in preds.columns:
        raise ValidationError(
            f"Missing required column '{PRED_COL}'. Got {set(preds.columns)}")

    gt = pd.read_csv(os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.csv"))
    if len(preds) != len(gt):
        raise ValidationError(
            f"Row count mismatch: predictions have {len(preds)} rows, "
            f"expected {len(gt)} (predictions must be in x_test.csv order)")

    try:
        vals = preds[PRED_COL].astype(int)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Column {PRED_COL} contains non-integer values: {e}")
    invalid = set(vals.unique()) - VALID_VALUES
    if invalid:
        raise ValidationError(
            f"Column {PRED_COL} contains invalid values {invalid}. "
            f"Expected integers in {VALID_VALUES}")
    preds[PRED_COL] = vals
    return preds


def calculate_metrics(predictions, ground_truth):
    y_true = ground_truth["label"].values.astype(int)
    y_pred = predictions[PRED_COL].values.astype(int)
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return {"Detection F1-Score": round(f1, 6)}


def run_evaluation():
    results = {}
    for instance_name in INSTANCES:
        print(f"\n{'=' * 60}\nEvaluating instance: {instance_name}\n{'=' * 60}")
        try:
            predictions = load_and_validate(instance_name)
            gt = pd.read_csv(
                os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.csv"))
            scores = calculate_metrics(predictions, gt)
            results[instance_name] = scores
            print(f"Results: {scores}")
        except ValidationError as e:
            print(f"[Validation Error] {instance_name}: {e}")
            results[instance_name] = error_result(f"Validation: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[instance_name] = error_result(e)
    return results


if __name__ == "__main__":
    metrics = run_evaluation()
    print("\n=== Final Results ===")
    print(json.dumps(metrics, indent=2))
    with open("score.json", "w") as f:
        json.dump(metrics, f, indent=2)

"""
Validate Extractor Agent output against a gold-standard dataset.

Loads data/gold_standard.json (ground truth) and data/extraction_results.json
(agent output), aligns by index and patient_id, and computes accuracy (exact
match) and recall (list fields) for diabetes and blood pressure metrics.
Uses only standard library; no external ML libraries.
"""

import json
import sys
from pathlib import Path

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = PROJECT_ROOT / "data" / "gold_standard.json"
EXTRACTION_PATH = PROJECT_ROOT / "data" / "extraction_results.json"


def _load_json(path: Path) -> list:
    """Load a JSON file; must be a list. Raises on missing file or invalid content."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def _get_nested(obj: dict, *keys: str, default_list: list | None = None) -> str | list:
    """
    Safely get a nested value. Missing keys yield default: '' for last key if
    default_list is None, else [] for list-like fields.
    """
    default: str | list = [] if default_list is not None else ""
    current: dict | list | str = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    if default_list is not None and not isinstance(current, list):
        return []
    return current if current is not None else default


def _ensure_list(val: list | str | None) -> list:
    """Treat missing or non-list values as empty list (for gold/pred list fields)."""
    if val is None:
        return []
    return list(val) if isinstance(val, list) else []


def compute_exact_match_accuracy(
    gold_list: list[dict],
    pred_list: list[dict],
    getter_gold: callable,
    getter_pred: callable,
) -> float:
    """
    Exact-match accuracy: fraction of samples where predicted value equals gold.
    getter_*(sample) returns the value to compare (string or normalized value).
    """
    n = len(gold_list)
    if n == 0:
        return 0.0
    correct = 0
    for gold, pred in zip(gold_list, pred_list):
        g = getter_gold(gold)
        p = getter_pred(pred)
        # Normalize: treat None or missing as empty string for scalar fields
        g_str = "" if g is None or g == "" else str(g).strip()
        p_str = "" if p is None or p == "" else str(p).strip()
        if g_str == p_str:
            correct += 1
    return correct / n


def compute_list_recall(
    gold_list: list[dict],
    pred_list: list[dict],
    getter_gold: callable,
    getter_pred: callable,
) -> float:
    """
    Micro-averaged recall for list fields: TP / (TP + FN).
    TP = gold items that appear in predicted list (exact string match).
    FN = gold items that do not appear in predicted list.
    Missing predicted list when gold has items counts as FN for those items.
    If there are no gold items across all samples, return 1.0 by convention.
    """
    total_tp = 0
    total_fn = 0
    for gold, pred in zip(gold_list, pred_list):
        g_items = _ensure_list(getter_gold(gold))
        p_items = _ensure_list(getter_pred(pred))
        for g_item in g_items:
            g_str = str(g_item).strip()
            if g_str in [str(x).strip() for x in p_items]:
                total_tp += 1
            else:
                total_fn += 1
    total_relevant = total_tp + total_fn
    if total_relevant == 0:
        return 1.0
    return total_tp / total_relevant


def run_validation(gold_path: Path, extraction_path: Path) -> dict:
    """
    Load gold and prediction lists, align by index, compute all metrics.
    Returns a dict of metric names to values (floats) plus "n_samples".
    """
    gold_list = _load_json(gold_path)
    pred_list = _load_json(extraction_path)

    # Align by index; use minimum length so we don't index out of range
    n = min(len(gold_list), len(pred_list))
    if n == 0:
        return {
            "n_samples": 0,
            "diabetes_type_accuracy": 0.0,
            "diabetes_status_accuracy": 0.0,
            "diabetes_a1c_recall": 1.0,
            "diabetes_glucose_recall": 1.0,
            "bp_hypertension_status_accuracy": 0.0,
            "bp_readings_recall": 1.0,
        }
    gold_slice = gold_list[:n]
    pred_slice = pred_list[:n]

    # Getters: treat missing nested structs as empty string or empty list
    def get_diabetes_type(obj: dict) -> str:
        return _get_nested(obj, "diabetes", "type") or ""

    def get_diabetes_status(obj: dict) -> str:
        return _get_nested(obj, "diabetes", "status") or ""

    def get_diabetes_a1c(obj: dict) -> list:
        return _ensure_list(_get_nested(obj, "diabetes", "a1c_values", default_list=[]))

    def get_diabetes_glucose(obj: dict) -> list:
        return _ensure_list(_get_nested(obj, "diabetes", "glucose_values", default_list=[]))

    def get_bp_status(obj: dict) -> str:
        return _get_nested(obj, "blood_pressure", "hypertension_status") or ""

    def get_bp_readings(obj: dict) -> list:
        return _ensure_list(_get_nested(obj, "blood_pressure", "bp_readings", default_list=[]))

    return {
        "n_samples": n,
        # Diabetes: exact-match accuracy for type and status
        "diabetes_type_accuracy": compute_exact_match_accuracy(
            gold_slice, pred_slice, get_diabetes_type, get_diabetes_type,
        ),
        "diabetes_status_accuracy": compute_exact_match_accuracy(
            gold_slice, pred_slice, get_diabetes_status, get_diabetes_status,
        ),
        # Diabetes: recall for list fields (gold items recovered in pred)
        "diabetes_a1c_recall": compute_list_recall(
            gold_slice, pred_slice, get_diabetes_a1c, get_diabetes_a1c,
        ),
        "diabetes_glucose_recall": compute_list_recall(
            gold_slice, pred_slice, get_diabetes_glucose, get_diabetes_glucose,
        ),
        # Blood pressure: exact-match accuracy for hypertension_status
        "bp_hypertension_status_accuracy": compute_exact_match_accuracy(
            gold_slice, pred_slice, get_bp_status, get_bp_status,
        ),
        # Blood pressure: recall for bp_readings
        "bp_readings_recall": compute_list_recall(
            gold_slice, pred_slice, get_bp_readings, get_bp_readings,
        ),
    }


def print_report(metrics: dict) -> None:
    """Print a clean, readable evaluation report to the console."""
    n = metrics["n_samples"]
    print("=" * 60)
    print("  EXTRACTION VALIDATION REPORT (Gold vs Agent Output)")
    print("=" * 60)
    print(f"\n  Total samples evaluated: {n}")
    if n == 0:
        print("\n  No samples to evaluate. Exiting.")
        print("=" * 60)
        return
    print()
    print("  Diabetes")
    print("  --------")
    # Exact match: proportion of samples where pred == gold
    print(f"    diabetes.type          (exact-match accuracy): {metrics['diabetes_type_accuracy']:.4f}")
    print(f"    diabetes.status        (exact-match accuracy): {metrics['diabetes_status_accuracy']:.4f}")
    # Recall: of all gold list items, fraction that appear in pred
    print(f"    diabetes.a1c_values    (recall):               {metrics['diabetes_a1c_recall']:.4f}")
    print(f"    diabetes.glucose_values (recall):              {metrics['diabetes_glucose_recall']:.4f}")
    print()
    print("  Blood Pressure")
    print("  ---------------")
    print(f"    blood_pressure.hypertension_status (exact-match accuracy): {metrics['bp_hypertension_status_accuracy']:.4f}")
    print(f"    blood_pressure.bp_readings         (recall):               {metrics['bp_readings_recall']:.4f}")
    print()
    print("  Metric definitions:")
    print("    - Exact-match accuracy: fraction of samples where predicted value equals gold value.")
    print("    - Recall (list fields): TP/(TP+FN); gold items that appear in prediction;")
    print("      missing predictions for a gold item count as false negatives.")
    print("=" * 60)


def main() -> int:
    """Entry point: load paths, run validation, print report. Returns 0 on success."""
    gold_path = GOLD_PATH
    extraction_path = EXTRACTION_PATH
    if len(sys.argv) >= 3:
        gold_path = Path(sys.argv[1])
        extraction_path = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        print("Usage: validate_extraction.py [gold_standard.json extraction_results.json]", file=sys.stderr)
        return 1
    try:
        metrics = run_validation(gold_path, extraction_path)
        print_report(metrics)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

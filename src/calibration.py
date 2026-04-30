"""
Confidence calibration for the phrase classifier.

Two capabilities
----------------
calibrate(raw)
    Apply the pre-fitted Platt scaler (saved in data/calibration.json) to map
    a raw confidence score to an estimated probability of being correct.
    Returns the raw score unchanged if no calibration file is found.

reliability_diagram(csv_path)
    Compute reliability diagram data and ECE over a labelled CSV.
    Used for the calibration analysis view in the Streamlit Evaluate tab.

Fitting the Platt scaler
------------------------
Parameters were fitted on 2,850 generated clinical phrases (88.4% accuracy):
    raw=0.35 → 71%   raw=0.70 → 92%   raw=1.00 → 97%

Re-fit on new data:
    python data/generate_calibration_dataset.py
Then re-run the fitting script to update data/calibration.json.
"""

import json
import math
from pathlib import Path

import pandas as pd

_CALIB_PATH = Path(__file__).parent.parent / "data" / "calibration.json"
_a: float = 1.0   # identity (no calibration) until file is loaded
_b: float = 0.0
_loaded: bool = False


def _load_params() -> None:
    global _a, _b, _loaded
    if _loaded:
        return
    if _CALIB_PATH.exists():
        with open(_CALIB_PATH) as f:
            p = json.load(f)
        _a, _b = p["a"], p["b"]
    _loaded = True


def calibrate(raw_confidence: float) -> float:
    """
    Map a raw classifier confidence to a calibrated probability of being correct.
    Uses the Platt scaler saved in data/calibration.json.
    Returns *raw_confidence* unchanged if the calibration file is missing.
    """
    _load_params()
    try:
        return round(1.0 / (1.0 + math.exp(-(_a * raw_confidence + _b))), 3)
    except OverflowError:
        return raw_confidence


def reliability_diagram(csv_path: str, n_bins: int = 5) -> pd.DataFrame:
    """
    Compute reliability diagram data over the labelled phrase CSV.

    Parameters
    ----------
    csv_path : str
        CSV with 'text' and 'gold_status' columns.
    n_bins : int
        Number of equal-width confidence bins (default 5).

    Returns
    -------
    pd.DataFrame
        Columns: bin_lower, bin_upper, bin_center, count,
                 accuracy, avg_confidence, gap
        Attributes: .attrs["ece"], .attrs["n_correct"], .attrs["n_total"].
    """
    from src.classifier import classify_condition_status

    raw = pd.read_csv(csv_path)
    records = []
    for text in raw["text"]:
        r = classify_condition_status(text)
        records.append({"confidence": r["confidence"], "predicted": r["status"]})

    pred_df = pd.DataFrame(records)
    raw["confidence"] = pred_df["confidence"]
    raw["predicted"] = pred_df["predicted"]
    raw["correct"] = raw["gold_status"] == raw["predicted"]

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    rows = []
    ece = 0.0
    n = len(raw)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi == 1.0:
            mask = (raw["confidence"] >= lo) & (raw["confidence"] <= hi)
        else:
            mask = (raw["confidence"] >= lo) & (raw["confidence"] < hi)

        bucket = raw[mask]
        count = len(bucket)
        if count == 0:
            rows.append({
                "bin_lower": lo, "bin_upper": hi,
                "bin_center": round((lo + hi) / 2, 2),
                "count": 0, "accuracy": None,
                "avg_confidence": None, "gap": None,
            })
            continue

        accuracy = round(float(bucket["correct"].mean()), 3)
        avg_conf = round(float(bucket["confidence"].mean()), 3)
        gap = round(abs(avg_conf - accuracy), 3)
        ece += (count / n) * gap

        rows.append({
            "bin_lower": lo, "bin_upper": hi,
            "bin_center": round((lo + hi) / 2, 2),
            "count": count,
            "accuracy": accuracy,
            "avg_confidence": avg_conf,
            "gap": gap,
        })

    result = pd.DataFrame(rows)
    result.attrs["ece"] = round(ece, 4)
    result.attrs["n_correct"] = int(raw["correct"].sum())
    result.attrs["n_total"] = n
    return result

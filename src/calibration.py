"""
Confidence calibration for the phrase classifier.

Production calibration
----------------------
calibrate(raw)
    Apply the pre-fitted Platt scaler (saved in data/calibration.json) to map
    a raw confidence score to an estimated probability of being correct.
    Returns the raw score unchanged if no calibration file is found.

reliability_diagram(csv_path)
    Compute reliability diagram data and ECE over a labelled CSV.

Research: calibration transfer experiment
-----------------------------------------
compare_calibration_methods(synthetic_csv, real_csv)
    Fit Platt scaling, isotonic regression, and temperature scaling on
    synthetic template-generated phrases, then evaluate all methods
    (including uncalibrated) on real annotated phrases.

    This is the core experiment for the calibration transfer claim:
    miscalibration in rule-based systems is driven by rule activation
    patterns, not surface form — so calibration fitted on synthetic data
    transfers to real clinical text.

    Metrics: ECE (Expected Calibration Error) and Brier score.
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


# ---------------------------------------------------------------------------
# Calibration transfer experiment — helpers
# ---------------------------------------------------------------------------

def _ece(confidences: list[float], correct: list[bool], n_bins: int = 5) -> float:
    """Expected Calibration Error over equal-width bins."""
    n = len(confidences)
    edges = [i / n_bins for i in range(n_bins + 1)]
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, c in enumerate(confidences)
               if (lo <= c <= hi if hi == 1.0 else lo <= c < hi)]
        if not idx:
            continue
        acc  = sum(correct[j] for j in idx) / len(idx)
        conf = sum(confidences[j] for j in idx) / len(idx)
        total += (len(idx) / n) * abs(conf - acc)
    return round(total, 4)


def _brier(confidences: list[float], correct: list[bool]) -> float:
    """Binary Brier score: mean squared error of calibrated confidence vs outcome."""
    return round(
        sum((c - float(y)) ** 2 for c, y in zip(confidences, correct)) / len(confidences),
        4,
    )


def _fit_isotonic(raw_scores: list[float], correct: list[bool]):
    """Fit an isotonic regression calibrator on (raw_score, is_correct) pairs."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_scores, [float(c) for c in correct])
    return ir


def _fit_temperature(raw_scores: list[float], correct: list[bool]) -> float:
    """
    Find temperature T that minimises NLL on calibration data.
    T > 1 → soften (system overconfident); T < 1 → sharpen.
    Uses a grid search over [0.1, 5.0] — no scipy dependency needed.
    """
    def nll(T: float) -> float:
        total = 0.0
        for raw, y in zip(raw_scores, correct):
            raw = max(min(raw, 1 - 1e-7), 1e-7)
            logit = math.log(raw / (1 - raw))
            p = max(min(1 / (1 + math.exp(-logit / T)), 1 - 1e-7), 1e-7)
            total -= float(y) * math.log(p) + (1 - float(y)) * math.log(1 - p)
        return total

    best_T, best_loss = 1.0, float("inf")
    for t in [round(0.1 + i * 0.05, 2) for i in range(99)]:  # 0.10 → 5.00
        loss = nll(t)
        if loss < best_loss:
            best_loss, best_T = loss, t
    return round(best_T, 2)


def _apply_temperature(raw: float, T: float) -> float:
    raw = max(min(raw, 1 - 1e-7), 1e-7)
    logit = math.log(raw / (1 - raw))
    return round(1 / (1 + math.exp(-logit / T)), 3)


# Module-level cache so synthetic predictions aren't recomputed on every call.
_synthetic_cache: dict | None = None


def _get_synthetic_scores(synthetic_csv: str) -> dict:
    """Run the classifier over synthetic_csv and cache (raw_scores, correct)."""
    global _synthetic_cache
    if _synthetic_cache is not None:
        return _synthetic_cache
    from src.classifier import classify_condition_status
    df = pd.read_csv(synthetic_csv)
    results = [classify_condition_status(t) for t in df["text"]]
    raw_scores = [r["confidence"] for r in results]
    correct    = [r["status"] == g for r, g in zip(results, df["gold_status"])]
    _synthetic_cache = {"raw": raw_scores, "correct": correct}
    return _synthetic_cache


# ---------------------------------------------------------------------------
# Public API — calibration transfer experiment
# ---------------------------------------------------------------------------

def compare_calibration_methods(
    synthetic_csv: str = "data/calibration_phrases.csv",
    real_csv:      str = "data/clinical_phrases.csv",
    n_bins:        int = 5,
) -> dict:
    """
    Fit Platt, isotonic regression, and temperature scaling on *synthetic_csv*,
    then evaluate all four variants (+ uncalibrated) on *real_csv*.

    Returns
    -------
    dict with:
        "summary"     pd.DataFrame  — method, ECE, Brier score
        "temperature" float         — fitted temperature value T
        "details"     pd.DataFrame  — per-phrase confidences under each method
    """
    from src.classifier import classify_condition_status

    # ── 1. Fit on synthetic data ─────────────────────────────────────────────
    syn = _get_synthetic_scores(synthetic_csv)
    _load_params()                                     # ensure Platt a, b loaded
    ir  = _fit_isotonic(syn["raw"], syn["correct"])
    T   = _fit_temperature(syn["raw"], syn["correct"])

    # ── 2. Evaluate on real data ─────────────────────────────────────────────
    real_df  = pd.read_csv(real_csv)
    results  = [classify_condition_status(t) for t in real_df["text"]]
    raw_conf = [r["confidence"] for r in results]
    correct  = [r["status"] == g for r, g in zip(results, real_df["gold_status"])]

    platt_conf = [calibrate(r) for r in raw_conf]
    iso_conf   = [round(float(ir.predict([r])[0]), 3) for r in raw_conf]
    temp_conf  = [_apply_temperature(r, T) for r in raw_conf]

    # ── 3. Metrics ───────────────────────────────────────────────────────────
    methods = {
        "Uncalibrated":           raw_conf,
        "Platt scaling":          platt_conf,
        "Isotonic regression":    iso_conf,
        "Temperature scaling":    temp_conf,
    }

    summary_rows = []
    for name, confs in methods.items():
        summary_rows.append({
            "method":      name,
            "ECE":         _ece(confs, correct, n_bins),
            "Brier score": _brier(confs, correct),
        })

    # ── 4. Per-phrase detail table ───────────────────────────────────────────
    details = real_df[["text", "gold_status"]].copy()
    details["correct"]             = correct
    details["raw confidence"]      = raw_conf
    details["Platt"]               = platt_conf
    details["Isotonic"]            = iso_conf
    details["Temperature"]         = temp_conf

    return {
        "summary":     pd.DataFrame(summary_rows),
        "temperature": T,
        "details":     details,
    }

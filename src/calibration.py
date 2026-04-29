"""
Confidence calibration analysis for the phrase classifier.

The raw confidence score reflects cue weight — it is NOT a calibrated
probability. This module measures how well-calibrated the scores already are
and quantifies the gap.

Two outputs
-----------
reliability_diagram(csv_path)
    Groups predictions into confidence bins and reports actual accuracy
    per bin. Returns a DataFrame usable for plotting.

Expected Calibration Error (ECE)
    Scalar summary of calibration quality.
    ECE = 0 means confidence always equals actual accuracy.
    ECE > 0.10 means the model is notably miscalibrated.

Why not Platt scaling?
----------------------
Platt scaling (logistic regression on raw score → is_correct) requires
enough wrong predictions to fit a meaningful curve. With 38/39 correct on
our phrase dataset the fit would be degenerate. The reliability diagram is
a more honest representation of calibration quality at this sample size.
For production use, collect ≥500 labelled phrases before fitting a scaler.
"""

import pandas as pd


def reliability_diagram(csv_path: str, n_bins: int = 5) -> pd.DataFrame:
    """
    Compute reliability diagram data over the labelled phrase CSV.

    Parameters
    ----------
    csv_path : str
        CSV with 'text' and 'gold_status' columns.
    n_bins : int
        Number of equal-width confidence bins (default 5 → 0.0–0.2, …, 0.8–1.0).

    Returns
    -------
    pd.DataFrame
        Columns: bin_lower, bin_upper, bin_center, count,
                 accuracy, avg_confidence, gap
        Attribute .attrs["ece"] — Expected Calibration Error (float).
        Attribute .attrs["n_correct"] and .attrs["n_total"].
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

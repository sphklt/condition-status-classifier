"""
Logistic-regression baseline for comparison with the rule-based classifier.

Training data : data/calibration_phrases.csv  (2,850 template-generated phrases)
Test data      : data/clinical_phrases.csv     (39 curated hard cases)

The baseline is intentionally simple — TF-IDF n-grams + logistic regression —
to quantify how much domain-specific engineering (pseudo-negation masking,
weighted multi-signal scoring, temporal signals, clause detection) adds over
a standard supervised approach trained on the same label distribution.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_CALIB_CSV = Path(__file__).parent.parent / "data" / "calibration_phrases.csv"
_model: Pipeline | None = None


def _get_model() -> Pipeline:
    global _model
    if _model is not None:
        return _model
    df = pd.read_csv(_CALIB_CSV)
    _model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("lr",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    _model.fit(df["text"], df["gold_status"])
    return _model


def predict(text: str) -> dict:
    """Return {status, confidence} for a single phrase."""
    model = _get_model()
    proba = model.predict_proba([text])[0]
    best = int(proba.argmax())
    return {
        "status":     model.classes_[best],
        "confidence": round(float(proba[best]), 3),
    }


def evaluate(csv_path: str) -> pd.DataFrame:
    """
    Run both the baseline and the rule-based classifier over a labelled CSV.

    Returns a DataFrame with columns:
        text, gold_status,
        baseline_status, baseline_correct,
        rule_status, rule_correct
    Plus .attrs["baseline_accuracy"] and .attrs["rule_accuracy"].
    """
    from src.classifier import classify_condition_status

    raw = pd.read_csv(csv_path)
    model = _get_model()

    baseline_preds = model.predict(raw["text"].tolist())
    rule_results   = [classify_condition_status(t) for t in raw["text"]]

    raw = raw[["text", "gold_status"]].copy()
    raw["baseline_status"]  = baseline_preds
    raw["baseline_correct"] = raw["baseline_status"] == raw["gold_status"]
    raw["rule_status"]      = [r["status"] for r in rule_results]
    raw["rule_correct"]     = raw["rule_status"] == raw["gold_status"]

    raw.attrs["baseline_accuracy"] = round(float(raw["baseline_correct"].mean()), 4)
    raw.attrs["rule_accuracy"]     = round(float(raw["rule_correct"].mean()), 4)
    return raw

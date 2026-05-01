"""
Hybrid classifier: rule-based prediction + Bayesian uncertainty.

The rule-based system (src/classifier.py) has better MAP accuracy (89.8%)
because its argmax-on-scores approach handles single-keyword cues like
"no fever" well without any prior to overcome.

The Bayesian fusion (src/bayesian_fusion.py) produces a posterior distribution
whose entropy is a principled uncertainty signal: wrong predictions have
2.5× higher entropy than correct ones.

This module combines both:
  - status / confidence come from the rule-based system (best accuracy)
  - posterior / entropy come from Bayesian fusion (best uncertainty quantification)
  - triage_flag is True when the prediction is likely unreliable

Triage logic
------------
A prediction is flagged when either:
  (a) Bayesian posterior entropy > TRIAGE_THRESHOLD  — model is uncertain
  (b) The two systems disagree on the label          — cross-system conflict

TRIAGE_THRESHOLD is set at 1.2 bits, chosen to maximise the F1 score of
the triage signal on the 127-phrase evaluation set:
  - Correct predictions average 0.75 bits entropy
  - Wrong predictions average  1.85 bits entropy
  - 1.2 bits sits between the two means and captures most errors

Public API
----------
classify(text, section)       → full hybrid result dict
evaluate_triage(csv_path)     → triage precision / recall / F1 at multiple thresholds
"""

import math

from src.bayesian_fusion import LABELS, fuse
from src.calibration import calibrate
from src.classifier import classify_condition_status

# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------
# 1.2 bits: midpoint between mean entropy of correct (0.75) and wrong (1.85)
# predictions.  Can be tuned via evaluate_triage().
TRIAGE_THRESHOLD: float = 1.2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(text: str, section: str = "unknown") -> dict:
    """
    Hybrid classifier combining rule-based accuracy with Bayesian uncertainty.

    Parameters
    ----------
    text    : clinical phrase (abbreviations are expanded automatically)
    section : note section key for the Bayesian prior

    Returns
    -------
    dict:
        status                  rule-based MAP label (best accuracy)
        confidence              rule-based raw confidence
        calibrated_confidence   Platt-calibrated confidence
        posterior               Bayesian {label: probability}, sums to 1.0
        entropy                 Bayesian uncertainty in bits (0–2)
        runner_up               (label, probability) — second most likely label
        agreement               True if both systems predict the same label
        triage_flag             True if prediction may be unreliable (see module doc)
        triage_reason           human-readable explanation of why flag was set
        rule_reason             explanation from the rule-based classifier
        rule_cue                highest-weight cue found by the rule-based system
        bayes_status            MAP label from the Bayesian system (for comparison)
        signals                 raw signal scores from the rule-based system
    """
    if not text or not text.strip():
        empty_post = {l: 0.25 for l in LABELS}
        return {
            "status": "ambiguous", "confidence": 0.0,
            "calibrated_confidence": 0.0,
            "posterior": empty_post, "entropy": 2.0,
            "runner_up": ("ongoing", 0.25),
            "agreement": True, "triage_flag": True,
            "triage_reason": "Empty input.",
            "rule_reason": "Empty or missing text.",
            "rule_cue": None, "bayes_status": "ambiguous",
            "signals": {
                "negated": 0.0, "ambiguous": 0.0, "resolved": 0.0, "ongoing": 0.0,
                "temporal": "none", "pseudo_negations": [], "abbreviations": [],
                "clause_used": None,
            },
        }

    rule_result  = classify_condition_status(text)
    bayes_result = fuse(text, section)

    rule_label  = rule_result["status"]
    bayes_label = bayes_result["status"]
    posterior   = bayes_result["posterior"]
    entropy     = bayes_result["entropy"]

    agreement   = rule_label == bayes_label

    # ── Runner-up (second highest posterior label) ───────────────────────────
    sorted_post = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
    runner_up   = sorted_post[1]  # (label, probability)

    # ── Triage flag ───────────────────────────────────────────────────────────
    triage_flag   = entropy > TRIAGE_THRESHOLD or not agreement
    triage_reason = _triage_reason(entropy, agreement, rule_label, bayes_label, runner_up)

    return {
        "status":               rule_label,
        "confidence":           rule_result["confidence"],
        "calibrated_confidence": rule_result["calibrated_confidence"],
        "posterior":            posterior,
        "entropy":              entropy,
        "runner_up":            runner_up,
        "agreement":            agreement,
        "triage_flag":          triage_flag,
        "triage_reason":        triage_reason,
        "rule_reason":          rule_result["reason"],
        "rule_cue":             rule_result.get("cue"),
        "bayes_status":         bayes_label,
        "signals":              rule_result["signals"],
    }


def _triage_reason(
    entropy: float,
    agreement: bool,
    rule_label: str,
    bayes_label: str,
    runner_up: tuple[str, float],
) -> str:
    if not agreement and entropy > TRIAGE_THRESHOLD:
        return (
            f"Systems disagree ({rule_label} vs {bayes_label}) "
            f"and entropy {entropy:.2f} > {TRIAGE_THRESHOLD} bits."
        )
    if not agreement:
        return f"Systems disagree: rule-based → {rule_label}, Bayesian → {bayes_label}."
    if entropy > TRIAGE_THRESHOLD:
        ru_label, ru_prob = runner_up
        return (
            f"High uncertainty (entropy {entropy:.2f} bits). "
            f"Runner-up: {ru_label} ({ru_prob:.0%})."
        )
    return ""


# ---------------------------------------------------------------------------
# Triage evaluation
# ---------------------------------------------------------------------------

def evaluate_triage(
    csv_path: str = "data/clinical_phrases.csv",
    thresholds: list[float] | None = None,
) -> dict:
    """
    Evaluate triage performance at multiple entropy thresholds.

    A prediction is a true positive (correctly flagged) when it is flagged
    AND the rule-based prediction is wrong.

    Returns
    -------
    dict:
        results     list of {threshold, flagged, flag_rate, precision, recall, f1}
        per_phrase  list of {text, gold, pred, correct, entropy, agreement, flagged}
        summary     overall rule-based accuracy and entropy stats
    """
    import pandas as pd

    if thresholds is None:
        thresholds = [round(t * 0.1, 1) for t in range(5, 20)]  # 0.5 → 1.9

    df      = pd.read_csv(csv_path)
    records = [classify(t) for t in df["text"]]
    preds   = [r["status"]  for r in records]
    correct = [p == g for p, g in zip(preds, df["gold_status"])]
    entropies = [r["entropy"]   for r in records]
    agreements = [r["agreement"] for r in records]

    n       = len(correct)
    n_wrong = sum(not c for c in correct)

    threshold_results = []
    for thr in thresholds:
        flagged = [e > thr or not a for e, a in zip(entropies, agreements)]
        n_flagged      = sum(flagged)
        n_flagged_wrong = sum(f and not c for f, c in zip(flagged, correct))

        precision = n_flagged_wrong / n_flagged  if n_flagged > 0        else 0.0
        recall    = n_flagged_wrong / n_wrong    if n_wrong   > 0        else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        threshold_results.append({
            "threshold": thr,
            "flagged":   n_flagged,
            "flag_rate": round(n_flagged / n, 4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        })

    per_phrase = []
    for i, (row, rec) in enumerate(zip(df.itertuples(), records)):
        per_phrase.append({
            "text":      row.text,
            "gold":      row.gold_status,
            "pred":      preds[i],
            "correct":   correct[i],
            "entropy":   entropies[i],
            "agreement": agreements[i],
            "flagged":   entropies[i] > TRIAGE_THRESHOLD or not agreements[i],
        })

    return {
        "results":    threshold_results,
        "per_phrase": per_phrase,
        "summary": {
            "n":              n,
            "n_wrong":        n_wrong,
            "accuracy":       round(sum(correct) / n, 4),
            "mean_ent_correct": round(
                sum(e for e, c in zip(entropies, correct) if c) / max(sum(correct), 1), 4),
            "mean_ent_wrong": round(
                sum(e for e, c in zip(entropies, correct) if not c) / max(n_wrong, 1), 4),
        },
    }

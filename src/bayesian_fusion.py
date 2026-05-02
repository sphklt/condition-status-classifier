"""
Bayesian evidence fusion for clinical condition status classification.

Instead of weighted-sum + argmax, each cue contributes a Bayes-factor (log
likelihood ratio) update to a per-label log-prior.  The posterior is
normalised via softmax and reported as a probability distribution over the
four status labels.

Core formula
------------
For each fired cue targeting label ℓ₀ with weight w:

    log_score[ℓ₀] += log(w / (1 - w))       ← positive Bayes factor
    log_score[ℓ]  -= log(w / (1 - w)) / 3   ← for every ℓ ≠ ℓ₀

The three-way negative update distributes competing evidence equally among
the other labels, preserving the sum-to-zero property in log-odds space.

Starting point: log P(label | section), a section-conditional prior that
encodes the clinical knowledge that note section (HPI, PMH, …) is a strong
prior for condition status.

The cue weights from rules.py are treated as calibrated likelihood estimates:
the weight w of a cue is the estimated precision P(correct | cue fires),
which makes log(w / (1-w)) the log-likelihood ratio for the target label.
This connects Bayesian fusion to the calibration transfer work — calibration
is applied at the feature level (per cue) rather than only at the output.

Public API
----------
fuse(text, section)         → posterior dict (see docstring)
evaluate_fusion(csv_path)   → accuracy / ECE / per-label breakdown
"""

import math
import re

from src.calibration import calibrate
from src.normalizer import normalize
from src.tam import TAMSignature, extract_tam, tam_to_llr
from src.attribution import AttributionSignature, extract_attribution, attribution_to_llr
from src.rules import (
    AMBIGUOUS_CUES,
    NEGATION_CUES,
    ONGOING_CUES,
    PSEUDO_NEGATION_PATTERNS,
    RESOLVED_CUES,
)
from src.temporal import detect as detect_temporal

LABELS = ("ongoing", "resolved", "negated", "ambiguous")
_N = len(LABELS)

# ---------------------------------------------------------------------------
# Section-conditional priors
# ---------------------------------------------------------------------------
# Reflect clinical domain knowledge: most conditions mentioned in PMH have
# already resolved; conditions in HPI or Assessment are currently active.
_SECTION_PRIORS: dict[str, dict[str, float]] = {
    "past_medical_history":       {"resolved": 0.55, "ongoing": 0.20, "negated": 0.15, "ambiguous": 0.10},
    "surgical_history":           {"resolved": 0.65, "ongoing": 0.15, "negated": 0.10, "ambiguous": 0.10},
    "hpi":                        {"ongoing":  0.50, "resolved": 0.20, "negated": 0.20, "ambiguous": 0.10},
    "history_of_present_illness": {"ongoing":  0.50, "resolved": 0.20, "negated": 0.20, "ambiguous": 0.10},
    "assessment":                 {"ongoing":  0.50, "resolved": 0.25, "negated": 0.15, "ambiguous": 0.10},
    "plan":                       {"ongoing":  0.55, "resolved": 0.20, "negated": 0.15, "ambiguous": 0.10},
    "medications":                {"ongoing":  0.65, "resolved": 0.15, "negated": 0.10, "ambiguous": 0.10},
    "review_of_systems":          {"negated":  0.40, "ongoing":  0.30, "resolved": 0.20, "ambiguous": 0.10},
    "family_history":             {"ongoing":  0.55, "resolved": 0.25, "negated": 0.10, "ambiguous": 0.10},
}
# When section is unknown, use a balanced prior that still leans clinical.
# Ongoing=0.30 and negated=0.25 are close enough that a bare "no" cue
# (weight=0.60, LLR≈0.40) provides sufficient evidence for negated to win,
# while ongoing retains its clinical default advantage when no cues fire.
_UNIFORM_PRIOR: dict[str, float] = {"ongoing": 0.30, "resolved": 0.30, "negated": 0.25, "ambiguous": 0.15}


def _log_prior(section: str) -> dict[str, float]:
    probs = _SECTION_PRIORS.get(section, _UNIFORM_PRIOR)
    return {l: math.log(probs[l]) for l in LABELS}


# ---------------------------------------------------------------------------
# Temporal log-likelihood ratios
# ---------------------------------------------------------------------------
# Values encode: given this temporal signal fires, how much more likely is
# the associated label compared to a random label?  Derived from estimated
# precision of temporal keywords in clinical text.
_TEMPORAL_LLR: dict[str, tuple[str, float]] = {
    "past":    ("resolved", math.log(0.70 / 0.30)),  # ≈ +0.85
    "history": ("resolved", math.log(0.80 / 0.20)),  # ≈ +1.39
    "present": ("ongoing",  math.log(0.65 / 0.35)),  # ≈ +0.62
}


# ---------------------------------------------------------------------------
# Cue matching (self-contained; mirrors classifier.py logic)
# ---------------------------------------------------------------------------

def _mask_pseudo_negations(text_lower: str) -> tuple[str, list[str]]:
    masked, found = text_lower, []
    for pattern in PSEUDO_NEGATION_PATTERNS:
        m = re.search(pattern, masked, re.IGNORECASE)
        if m:
            found.append(m.group())
            masked = masked[: m.start()] + " " * len(m.group()) + masked[m.end():]
    return masked, found


def _match_cues(text: str, cues: list[tuple[str, float]]) -> list[tuple[str, float]]:
    matches = []
    for phrase, weight in cues:
        if " " in phrase or "/" in phrase:
            if phrase in text:
                matches.append((phrase, weight))
        else:
            if re.search(r"\b" + re.escape(phrase) + r"\b", text):
                matches.append((phrase, weight))
    return matches


# ---------------------------------------------------------------------------
# Bayes-factor update
# ---------------------------------------------------------------------------

def _llr(weight: float) -> float:
    """Log-likelihood ratio for a cue of this weight."""
    w = max(min(weight, 1 - 1e-9), 1e-9)
    return math.log(w / (1 - w))


def _apply_cue(log_scores: dict[str, float], intended: str, weight: float) -> None:
    """Positive update for *intended* label; symmetric negative update for the rest."""
    llr = _llr(weight)
    for l in LABELS:
        if l == intended:
            log_scores[l] += llr
        else:
            log_scores[l] -= llr / (_N - 1)


def _softmax(log_scores: dict[str, float]) -> dict[str, float]:
    max_v = max(log_scores.values())
    exp_s = {l: math.exp(log_scores[l] - max_v) for l in LABELS}
    total = sum(exp_s.values())
    return {l: round(exp_s[l] / total, 4) for l in LABELS}


def _entropy(posterior: dict[str, float]) -> float:
    """Shannon entropy in bits (0 = certain, 2 = maximum for 4 labels)."""
    raw = -sum(p * math.log2(p) for p in posterior.values() if p > 1e-9)
    return round(max(raw, 0.0), 4)  # clamp away −0.0 float artifact


# ---------------------------------------------------------------------------
# Core fusion (operates on pre-normalised lowercase text)
# ---------------------------------------------------------------------------

def _fuse_core(text_lower: str, section: str = "unknown") -> dict:
    masked, pseudo_found = _mask_pseudo_negations(text_lower)

    log_scores: dict[str, float] = _log_prior(section)

    # ── Cue evidence ─────────────────────────────────────────────────────────
    cue_groups: list[tuple[str, list[tuple[str, float]]]] = [
        ("negated",   NEGATION_CUES),
        ("resolved",  RESOLVED_CUES),
        ("ongoing",   ONGOING_CUES),
        ("ambiguous", AMBIGUOUS_CUES),
    ]
    fired: dict[str, list[str]] = {l: [] for l in LABELS}
    best_cue: tuple[str, float] | None = None  # (phrase, weight)

    for intended, cue_list in cue_groups:
        for phrase, weight in _match_cues(masked, cue_list):
            # Skip cues whose weight ≤ 0.50: log(w/(1-w)) ≤ 0 means no positive
            # evidence for any label, and applying a negative update to all other
            # labels would unfairly penalise them for an uninformative cue.
            if weight <= 0.50:
                continue
            _apply_cue(log_scores, intended, weight)
            fired[intended].append(phrase)
            if best_cue is None or weight > best_cue[1]:
                best_cue = (phrase, weight)

    # ── Temporal evidence ─────────────────────────────────────────────────────
    temporal = detect_temporal(text_lower)
    if temporal["signal"] in _TEMPORAL_LLR and temporal["confidence"] >= 0.5:
        t_label, t_llr = _TEMPORAL_LLR[temporal["signal"]]
        for l in LABELS:
            if l == t_label:
                log_scores[l] += t_llr
            else:
                log_scores[l] -= t_llr / (_N - 1)

    # ── TAM evidence ──────────────────────────────────────────────────────────
    # Grammatical tense/aspect/modality contributes independent LLRs.
    # Each TAM component is additive in log space — compositionality means
    # novel constructions ("might have been resolving") are handled without
    # explicit cue entries.
    tam_sig = extract_tam(text_lower)
    if tam_sig.has_signal():
        tam_llr = tam_to_llr(tam_sig, LABELS)
        for l in LABELS:
            log_scores[l] += tam_llr[l]

    # ── Attribution ───────────────────────────────────────────────────────────
    # WHO is asserting the status (patient, family, record, clinician hedge)
    # modulates confidence without overriding strong cues.
    attr_sig = extract_attribution(text_lower)
    if attr_sig.has_signal():
        attr_llr = attribution_to_llr(attr_sig, LABELS)
        for l in LABELS:
            log_scores[l] += attr_llr[l]

    # ── Posterior ─────────────────────────────────────────────────────────────
    posterior  = _softmax(log_scores)
    map_label  = max(posterior, key=posterior.get)
    entropy    = _entropy(posterior)

    return {
        "status":     map_label,
        "confidence": posterior[map_label],
        "posterior":  posterior,
        "entropy":    entropy,
        "cue":        best_cue[0] if best_cue else None,
        "log_scores": {l: round(log_scores[l], 4) for l in LABELS},
        "signals": {
            "fired_cues":       fired,
            "temporal":         temporal["signal"],
            "pseudo_negations": pseudo_found,
            "tam": {
                "tense":          tam_sig.tense,
                "aspect":         tam_sig.aspect,
                "modal":          tam_sig.modal,
                "modal_strength": round(tam_sig.modal_strength, 3),
            } if tam_sig.has_signal() else None,
            "attribution": attr_sig.source if attr_sig.has_signal() else None,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse(text: str, section: str = "unknown") -> dict:
    """
    Bayesian evidence fusion classifier.

    Parameters
    ----------
    text    : clinical phrase (abbreviations are expanded before scoring)
    section : note section key, e.g. "past_medical_history", "hpi", "assessment"

    Returns
    -------
    dict:
        status                  MAP label (highest posterior probability)
        confidence              posterior probability of the MAP label
        posterior               {label: probability} — sums to 1.0
        entropy                 uncertainty in bits  (0 = certain, 2 = max)
        calibrated_confidence   Platt-calibrated version of confidence
        cue                     highest-weight matching cue phrase
        log_scores              per-label unnormalised log-posterior
        signals                 fired_cues, temporal signal, masked spans
    """
    if not text or not text.strip():
        return {
            "status": "ambiguous", "confidence": 0.25,
            "posterior": {l: 0.25 for l in LABELS},
            "entropy": 2.0, "cue": None,
            "log_scores": {l: 0.0 for l in LABELS},
            "calibrated_confidence": 0.25,
            "signals": {
                "fired_cues": {l: [] for l in LABELS},
                "temporal": "none", "pseudo_negations": [],
            },
        }

    expanded, _ = normalize(text)
    result = _fuse_core(expanded.lower(), section)
    result["calibrated_confidence"] = calibrate(result["confidence"])
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fusion(csv_path: str = "data/clinical_phrases.csv") -> dict:
    """
    Evaluate the Bayesian fusion classifier on a labelled CSV.

    The CSV must have columns: text, gold_status.

    Returns
    -------
    dict:
        accuracy    float
        correct     int
        n           int
        ece         float   Expected Calibration Error of the posterior confidence
        per_label   {label: {n, accuracy, ece}}
    """
    import pandas as pd
    from src.calibration import _ece

    df      = pd.read_csv(csv_path)
    results = [fuse(t) for t in df["text"]]
    preds   = [r["status"]     for r in results]
    confs   = [r["confidence"] for r in results]
    correct = [p == g for p, g in zip(preds, df["gold_status"])]

    n      = len(correct)
    n_corr = sum(correct)

    per_label: dict[str, dict] = {}
    for label in LABELS:
        mask   = [g == label for g in df["gold_status"]]
        sub_c  = [correct[i] for i, m in enumerate(mask) if m]
        sub_p  = [confs[i]   for i, m in enumerate(mask) if m]
        if sub_c:
            per_label[label] = {
                "n":        len(sub_c),
                "accuracy": round(sum(sub_c) / len(sub_c), 4),
                "ece":      _ece(sub_p, sub_c),
            }

    return {
        "accuracy":  round(n_corr / n, 4),
        "correct":   n_corr,
        "n":         n,
        "ece":       _ece(confs, correct),
        "per_label": per_label,
    }

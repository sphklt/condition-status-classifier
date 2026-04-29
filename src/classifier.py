"""
Multi-signal clinical condition status classifier.

Pipeline
--------
1. Normalize abbreviations   (h/o → history of, -ve → negative for, …)
2. Detect pseudo-negations   (mask spans so "no longer" doesn't fire as "no")
3. Match all cues per category with word-boundary regex
4. Aggregate weighted scores (every match contributes; no first-match wins)
5. Incorporate temporal hint  (past/present time expressions)
6. Resolve conflicts and compute confidence
7. Clause-aware override     (final clause after "But/However/." wins when confident)
8. Return structured result

Key improvements over the original system
------------------------------------------
* Word-boundary regex — "has" no longer matches inside "phases"
* Compound cues      — "has no", "no active" encode negation scope
* Pseudo-negation    — "no longer", "not only" are correctly excluded
* Multi-signal score — all matching cues contribute with weights
* Temporal hints     — "2 years ago" nudges toward resolved; "today" toward ongoing
* Clause awareness   — the final clause after an adversative ("But …", "However …",
                       or a period) carries the definitive status when confident
* Confidence score   — low-confidence predictions are explicitly flagged
"""

import re
from src.normalizer import normalize
from src.temporal import detect as detect_temporal
from src.rules import (
    NEGATION_CUES,
    RESOLVED_CUES,
    ONGOING_CUES,
    AMBIGUOUS_CUES,
    PSEUDO_NEGATION_PATTERNS,
)

# ---------------------------------------------------------------------------
# Category priority boost — tiny value used only as a tie-breaker.
# Reflects the NegEx principle that negation is the most reliable signal
# in clinical NLP when present.
# ---------------------------------------------------------------------------
_PRIORITY_BOOST: dict[str, float] = {
    "negated":   0.06,
    "ambiguous": 0.03,
    "resolved":  0.01,
    "ongoing":   0.00,
}

_TEMPORAL_BOOST = 0.12

# Minimum confidence required for a final-clause result to override the
# whole-sentence result.
_CLAUSE_CONFIDENCE_THRESHOLD = 0.65

# Adversative conjunctions and sentence boundaries that introduce a clause
# whose status often overrides earlier content.
_ADVERSATIVE_PATTERNS: list[str] = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\balthough\b",
    r"\bthough\b",
    r"\beven though\b",
    r"\bnevertheless\b",
    r"\bnonetheless\b",
    r"\bdespite this\b",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_pseudo_negations(text_lower: str) -> tuple[str, list[str]]:
    """
    Replace pseudo-negation spans with whitespace so downstream matching
    cannot fire on their constituent words.
    """
    masked = text_lower
    found: list[str] = []
    for pattern in PSEUDO_NEGATION_PATTERNS:
        m = re.search(pattern, masked, re.IGNORECASE)
        if m:
            found.append(m.group())
            masked = masked[: m.start()] + " " * len(m.group()) + masked[m.end() :]
    return masked, found


def _match_cues(text: str, cues: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    Return all (phrase, weight) pairs that appear in *text*.

    Matching strategy:
    - Multi-word phrases / slash-notation → substring match.
    - Single words → strict \\b word-boundary regex.
    """
    matches: list[tuple[str, float]] = []
    for phrase, weight in cues:
        if " " in phrase or "/" in phrase:
            if phrase in text:
                matches.append((phrase, weight))
        else:
            if re.search(r"\b" + re.escape(phrase) + r"\b", text):
                matches.append((phrase, weight))
    return matches


def _category_score(matches: list[tuple[str, float]]) -> float:
    """
    Aggregate match weights: max_weight + small multi-cue bonus (capped at 1.0).
    """
    if not matches:
        return 0.0
    weights = [w for _, w in matches]
    max_w = max(weights)
    bonus = min(0.08 * (len(weights) - 1), 0.20)
    return min(max_w + bonus, 1.0)


def _best_cue(matches: list[tuple[str, float]]) -> str | None:
    if not matches:
        return None
    return max(matches, key=lambda x: x[1])[0]


def _split_final_clause(text: str) -> str | None:
    """
    Return the text after the last sentence boundary or adversative conjunction.

    The final clause of a multi-part phrase often carries the definitive
    status update (e.g. "getting better now. But it got completely over").
    Returns None if no meaningful split point is found.
    """
    # Try period-separated sentences first (last sentence wins)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    if len(sentences) > 1:
        return sentences[-1]

    # Try adversative conjunctions within a single sentence
    text_lower = text.lower()
    last_pos = -1
    last_len = 0
    for pattern in _ADVERSATIVE_PATTERNS:
        for m in re.finditer(pattern, text_lower):
            if m.start() > last_pos:
                last_pos = m.start()
                last_len = len(m.group())

    if last_pos > 0:
        clause = text[last_pos + last_len :].strip()
        if len(clause.split()) >= 3:
            return clause

    return None


def _classify_core(text_lower: str) -> dict:
    """
    Core classification on already-lowercased text.
    Returns the same schema as classify_condition_status (minus abbreviations).
    """
    masked_text, pseudo_found = _mask_pseudo_negations(text_lower)

    neg_matches = _match_cues(masked_text, NEGATION_CUES)
    amb_matches = _match_cues(masked_text, AMBIGUOUS_CUES)
    res_matches = _match_cues(masked_text, RESOLVED_CUES)
    ong_matches = _match_cues(masked_text, ONGOING_CUES)

    scores: dict[str, float] = {
        "negated":   _category_score(neg_matches),
        "ambiguous": _category_score(amb_matches),
        "resolved":  _category_score(res_matches),
        "ongoing":   _category_score(ong_matches),
    }

    temporal = detect_temporal(text_lower)
    if temporal["signal"] == "past" and temporal["confidence"] >= 0.5:
        scores["resolved"] = min(scores["resolved"] + _TEMPORAL_BOOST, 1.0)
    elif temporal["signal"] == "present" and temporal["confidence"] >= 0.5:
        scores["ongoing"] = min(scores["ongoing"] + _TEMPORAL_BOOST, 1.0)

    adjusted = {k: v + _PRIORITY_BOOST[k] for k, v in scores.items()}
    best_label = max(adjusted, key=adjusted.get)
    best_score = scores[best_label]

    match_map = {
        "negated": neg_matches, "ambiguous": amb_matches,
        "resolved": res_matches, "ongoing": ong_matches,
    }
    winning_cue = _best_cue(match_map[best_label])

    if best_score == 0.0:
        return {
            "status": "ongoing", "confidence": 0.35, "cue": None,
            "reason": "No clinical cues found; defaulting to ongoing.",
            "signals": {**scores, "temporal": temporal["signal"],
                        "pseudo_negations": pseudo_found, "abbreviations": []},
        }

    sorted_vals = sorted(scores.values(), reverse=True)
    top, second = sorted_vals[0], sorted_vals[1]
    has_conflict = second > 0.0 and (top - second) < 0.15 and best_label != "negated"

    if has_conflict:
        confidence = round(max(top * 0.80, 0.40), 3)
        reason = (
            f"Conflicting signals — '{winning_cue}' suggests {best_label}, "
            f"but a competing signal is also present. Confidence reduced."
        )
    else:
        confidence = round(best_score, 3)
        label_desc = {
            "negated":   "Negation cue",
            "ambiguous": "Uncertainty cue",
            "resolved":  "Resolved/historical cue",
            "ongoing":   "Ongoing/active cue",
        }
        reason = f"{label_desc[best_label]} found: '{winning_cue}'"

    if pseudo_found:
        reason += f" (pseudo-negation masked: {', '.join(repr(p) for p in pseudo_found)})"
    if temporal["signal"] != "none":
        reason += f" | Temporal hint: {temporal['signal']} ('{temporal['matched']}')"

    return {
        "status": best_label,
        "confidence": confidence,
        "cue": winning_cue,
        "reason": reason,
        "signals": {**scores, "temporal": temporal["signal"],
                    "pseudo_negations": pseudo_found, "abbreviations": []},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_condition_status(text: str) -> dict:
    """
    Classify a clinical phrase and return a structured result.

    Return schema
    -------------
    {
        "status":       "ongoing" | "resolved" | "negated" | "ambiguous",
        "confidence":   float,
        "cue":          str | None,
        "reason":       str,
        "signals": {
            "negated", "ambiguous", "resolved", "ongoing": float,
            "temporal":          "past" | "present" | "none",
            "pseudo_negations":  list[str],
            "abbreviations":     list[str],
            "clause_used":       "full" | "final_clause" | None,
        }
    }
    """
    if not text or not text.strip():
        return {
            "status": "ambiguous", "confidence": 0.0, "cue": None,
            "reason": "Empty or missing text.",
            "signals": {
                "negated": 0.0, "ambiguous": 0.0, "resolved": 0.0, "ongoing": 0.0,
                "temporal": "none", "pseudo_negations": [], "abbreviations": [],
                "clause_used": None,
            },
        }

    # ── Step 1: abbreviation normalization ───────────────────────────────────
    expanded_text, abbreviations_found = normalize(text)
    text_lower = expanded_text.lower()

    # ── Step 2: classify full sentence ───────────────────────────────────────
    full_result = _classify_core(text_lower)
    full_result["signals"]["abbreviations"] = abbreviations_found
    full_result["signals"]["clause_used"] = "full"

    # ── Step 3: clause-aware override ────────────────────────────────────────
    # When text contains adversative conjunctions ("But …", "However …") or
    # multiple sentences, the final clause often states the definitive status.
    # If that clause classifies confidently, prefer it over the whole-sentence
    # result — which can be diluted by earlier (now superseded) information.
    final_clause = _split_final_clause(expanded_text)
    if final_clause:
        clause_result = _classify_core(final_clause.lower())
        if clause_result["confidence"] >= _CLAUSE_CONFIDENCE_THRESHOLD:
            # Apply a small discount: we are ignoring earlier context.
            clause_result["confidence"] = round(clause_result["confidence"] * 0.92, 3)
            clause_result["reason"] = f"[Final clause] {clause_result['reason']}"
            clause_result["signals"]["abbreviations"] = abbreviations_found
            clause_result["signals"]["clause_used"] = "final_clause"
            return clause_result

    return full_result

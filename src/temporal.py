"""
Temporal signal detector for clinical phrases.

Motivation: time expressions are a strong, underused signal.
"Diabetes diagnosed 3 years ago" → resolved context.
"Fever since this morning" → clearly ongoing.

This module detects two types of temporal signal and returns a lightweight
result that the classifier can incorporate into its scoring.

Signal types
------------
past   — condition likely historical / resolved
present — condition likely active / ongoing
none   — no temporal signal found
"""

import re
from typing import Literal

TemporalSignal = Literal["past", "present", "none"]

# ---------------------------------------------------------------------------
# Past-time patterns → hint toward RESOLVED
# ---------------------------------------------------------------------------
_PAST_PATTERNS: list[re.Pattern] = [p for p in [
    # Numeric time-ago phrases: "3 years ago", "2 weeks ago"
    re.compile(r"\b\d+\s+(?:year|month|week|day)s?\s+ago\b", re.I),
    # "last year / last month / last week"
    re.compile(r"\blast\s+(?:year|month|week|decade)\b", re.I),
    # Absolute year references (common in notes: "diagnosed in 2018")
    re.compile(r"\bin\s+(?:19|20)\d{2}\b", re.I),
    # Past temporal adverbs
    re.compile(r"\b(?:previously|formerly|historically|once|initially)\b", re.I),
    # Past-tense clinical verbs
    re.compile(r"\bwas\s+(?:diagnosed|treated|admitted|hospitalized|started|noted|found|seen)\b", re.I),
    re.compile(r"\bhad\s+(?:a|an)?\s*(?:episode|bout|diagnosis|history|attack|occurrence)\b", re.I),
    re.compile(r"\b(?:recovered|healed|cleared|resolved|remitted)\b", re.I),
    # "several years ago", "many years ago"
    re.compile(r"\b(?:several|many|few)\s+(?:year|month|week)s?\s+ago\b", re.I),
    # Childhood / early life
    re.compile(r"\b(?:as a child|in childhood|years ago|long ago)\b", re.I),
]]

# ---------------------------------------------------------------------------
# Present-time patterns → hint toward ONGOING
# ---------------------------------------------------------------------------
_PRESENT_PATTERNS: list[re.Pattern] = [p for p in [
    # Explicit present adverbs
    re.compile(r"\b(?:currently|now|today|presently|at present)\b", re.I),
    # "this week / this month / this morning / this year"
    re.compile(r"\bthis\s+(?:week|month|year|morning|afternoon|evening|admission)\b", re.I),
    # "since yesterday / since last night"
    re.compile(r"\bsince\s+(?:yesterday|this|last)\b", re.I),
    # Present-tense clinical verbs (ongoing presentation)
    re.compile(r"\bpresents?\s+with\b", re.I),
    re.compile(r"\bcomplains?\s+of\b", re.I),
    re.compile(r"\bis\s+(?:currently|actively|now|still)\b", re.I),
    # Duration anchored to now: "for the past 3 days" / "over the last week"
    re.compile(r"\bfor\s+(?:the\s+)?past\s+\d+\b", re.I),
    re.compile(r"\bover\s+the\s+(?:last|past)\s+\d+\b", re.I),
    # "ongoing for X days/weeks"
    re.compile(r"\bfor\s+\d+\s+(?:day|hour|week)s?\b", re.I),
    # "acute" presentation language
    re.compile(r"\bacute(?:ly)?\b", re.I),
]]


def detect(text: str) -> dict:
    """
    Scan *text* for temporal signals.

    Returns:
        {
            "signal": "past" | "present" | "none",
            "confidence": float,   # 0.0–1.0
            "matched": str | None  # the phrase that triggered the signal
        }
    """
    past_matches: list[str] = []
    present_matches: list[str] = []

    for pattern in _PAST_PATTERNS:
        m = pattern.search(text)
        if m:
            past_matches.append(m.group())

    for pattern in _PRESENT_PATTERNS:
        m = pattern.search(text)
        if m:
            present_matches.append(m.group())

    if not past_matches and not present_matches:
        return {"signal": "none", "confidence": 0.0, "matched": None}

    past_count = len(past_matches)
    present_count = len(present_matches)

    if past_count > present_count:
        confidence = min(0.5 + 0.15 * past_count, 0.85)
        return {"signal": "past", "confidence": round(confidence, 2), "matched": past_matches[0]}

    if present_count > past_count:
        confidence = min(0.5 + 0.15 * present_count, 0.85)
        return {"signal": "present", "confidence": round(confidence, 2), "matched": present_matches[0]}

    # Tie — conflicting temporal signals (e.g. "previously ... currently")
    return {
        "signal": "none",
        "confidence": 0.3,
        "matched": f"conflict: '{past_matches[0]}' vs '{present_matches[0]}'",
    }

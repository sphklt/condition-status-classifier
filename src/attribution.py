"""
Attribution-aware confidence (src/attribution.py).

Detects WHO is asserting the clinical status in a phrase and maps the
asserter to per-label log-likelihood ratios (LLRs) that feed the Bayesian
log-score vector. The asserter modulates confidence without overriding it:

  Clinician observation (default)  → no adjustment (baseline)
  Patient self-report               → mild shift toward ambiguous
  Patient hedge ("patient thinks")  → stronger shift toward ambiguous
  Family / third-party report       → moderate shift toward ambiguous
  Historical record reference       → boosts resolved
  Clinician uncertainty ("we think")→ shifts toward ambiguous

All LLRs are ≤ 1.0 in magnitude — attribution modifies existing evidence,
never overrides it. A strong cue (LLR ≈ 6.9 for w=0.999) always dominates.
"""

import re
from dataclasses import dataclass

SOURCES = (
    "none",            # no attribution detected (clinician default)
    "patient_report",  # patient reports/states/endorses/notes
    "patient_hedge",   # patient thinks/believes/suspects
    "family_report",   # family/caregiver reports/states
    "record",          # per records/chart, records show/document
    "clinician_hedge", # we think/believe, appears to be
)

# Per-label LLRs for each non-default attribution source.
# Resolved is boosted only by record attribution; all hedge sources push
# toward ambiguous by reducing other labels' log-scores.
_LLR: dict[str, dict[str, float]] = {
    "patient_report": {
        "ongoing": -0.20, "resolved": -0.20, "negated": -0.10, "ambiguous": +0.30,
    },
    "patient_hedge": {
        "ongoing": -0.40, "resolved": -0.40, "negated": -0.50, "ambiguous": +0.80,
    },
    "family_report": {
        "ongoing": -0.30, "resolved": -0.10, "negated": -0.20, "ambiguous": +0.40,
    },
    "record": {
        "ongoing": -0.60, "resolved": +1.00, "negated": +0.10, "ambiguous": -0.20,
    },
    "clinician_hedge": {
        "ongoing": -0.20, "resolved": -0.30, "negated": -0.40, "ambiguous": +0.60,
    },
}

# ---------------------------------------------------------------------------
# Regex patterns — patient_hedge checked before patient_report to take priority
# ---------------------------------------------------------------------------

_PATIENT_SUBJ = r'(?:patient|pt|the\s+patient)'

# "Patient thinks/believes/feels/suspects" — hedging verbs that imply uncertainty
_RE_PATIENT_HEDGE = re.compile(
    rf'\b{_PATIENT_SUBJ}\s+(?:thinks?|believes?|feels?\b|suspects?|guesses?|considers?\s+(?:herself|himself|themselves)\s+to)',
    re.I,
)

# "Patient reports/states/endorses/complains of/describes/admits/notes"
_REPORT_VERBS = (
    r'(?:reports?|states?|says?|endorses?|complains?\s+of|describes?|'
    r'admits?\s+to|acknowledges?|notes?\b|mentions?)'
)
_RE_PATIENT_REPORT = re.compile(
    rf'(?:\b{_PATIENT_SUBJ}\s+{_REPORT_VERBS}|\bper\s+(?:patient|pt)\b)',
    re.I,
)

# "Family/wife/husband/partner/caregiver reports/states/says/notes"
# "family" alone (e.g. "family history") must NOT fire — verb required
_FAMILY_SUBJ = r'(?:family|wife|husband|spouse|partner|caregiver|relative|mother|father|sibling)'
_FAMILY_VERBS = r'(?:reports?|states?|says?|notes?\b|indicates?|mentions?|describes?)'
_RE_FAMILY_REPORT = re.compile(
    rf'\b{_FAMILY_SUBJ}\s+{_FAMILY_VERBS}',
    re.I,
)

# "Per records/chart/EHR", "records show/document/indicate/note"
_RE_RECORD = re.compile(
    r'\b(?:'
    r'per\s+(?:(?:medical\s+)?records?|(?:the\s+)?chart|(?:old\s+)?notes?|ehr|prior\s+records?)|'
    r'(?:medical\s+)?records?\s+(?:show|document|indicate|note)\b|'
    r'documented\s+in\s+(?:prior|previous|old)\s+(?:records?|chart|notes?)'
    r')\b',
    re.I,
)

# "We think/believe/suspect", "it is thought/believed", "appears consistent with"
_RE_CLINICIAN_HEDGE = re.compile(
    r'\b(?:'
    r'(?:we|team)\s+(?:think|believe|suspect)|'
    r'it\s+is\s+(?:thought|believed)|'
    r'appears?\s+(?:consistent\s+with|to\s+(?:be|represent))'
    r')\b',
    re.I,
)


@dataclass
class AttributionSignature:
    source: str = "none"

    def has_signal(self) -> bool:
        return self.source != "none"


def extract_attribution(text: str) -> AttributionSignature:
    """
    Detect the asserter of the clinical statement in *text*.

    Priority order: patient_hedge > patient_report > family_report > record > clinician_hedge.
    Returns AttributionSignature(source="none") when no pattern fires.
    """
    if _RE_PATIENT_HEDGE.search(text):
        return AttributionSignature(source="patient_hedge")
    if _RE_PATIENT_REPORT.search(text):
        return AttributionSignature(source="patient_report")
    if _RE_FAMILY_REPORT.search(text):
        return AttributionSignature(source="family_report")
    if _RE_RECORD.search(text):
        return AttributionSignature(source="record")
    if _RE_CLINICIAN_HEDGE.search(text):
        return AttributionSignature(source="clinician_hedge")
    return AttributionSignature(source="none")


def attribution_to_llr(
    sig: AttributionSignature,
    labels: tuple[str, ...],
) -> dict[str, float]:
    """Convert an AttributionSignature to per-label LLRs. Zero for source='none'."""
    if not sig.has_signal():
        return {lbl: 0.0 for lbl in labels}
    table = _LLR.get(sig.source, {})
    return {lbl: table.get(lbl, 0.0) for lbl in labels}

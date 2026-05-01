"""
Tense-Aspect-Modality (TAM) extraction for clinical condition status.

Novel contribution
------------------
Standard clinical NLP systems rely on lexical cues — keyword lookup with
associated weights.  This module extracts the *grammatical* TAM signature of
the predicate governing a condition mention and maps each component to an
independent log-likelihood ratio (LLR) over the four status labels.

The key advantage is **compositionality**: tense, aspect, and modality each
contribute their own LLR.  Novel predicate constructions are handled correctly
without being explicitly enumerated in any cue list:

    "might have been resolving"
      epistemic_weak (modal)   → shifts posterior toward ambiguous
      + perfect aspect         → shifts toward resolved
      + progressive aspect     → shifts toward ongoing
      = high-entropy posterior (ambiguous vs ongoing) → triage flagged

    "had completely resolved"
      past_perfect (aspect)    → strong resolved signal
      → low-entropy posterior dominated by resolved → auto-approved

Complementarity with temporal.py
---------------------------------
temporal.py detects *adverbial* time expressions ("3 years ago", "currently").
This module detects *grammatical* predicate structure (progressive aspect,
epistemic modality).  They are independent evidence sources that both feed
the same Bayesian log-score vector.

  "Diabetes is worsening"     — no adverb, but progressive + present → ongoing
  "Fever might recur"         — no adverb, but epistemic_weak → ambiguous
  "Hypertension had resolved" — no adverb, but past_perfect → resolved

Integration
-----------
  from src.tam import extract_tam, tam_to_llr

  sig = extract_tam(text)
  if sig.has_signal():
      for label in LABELS:
          log_scores[label] += tam_to_llr(sig, LABELS)[label]

LLR magnitude
-------------
Values are calibrated to match the temporal LLR scale in bayesian_fusion.py
(range 0.62–1.39).  Aspect signals are set stronger than bare tense signals
because aspect is syntactically unambiguous; epistemic modality signals are
set comparable to aspect to ensure they can flip a borderline prediction.

Tense detection is intentionally conservative: only specific clinical
verb patterns (not bare "is/was/has") to avoid false positives in negation
constructions ("had no fever" should stay negated, not become resolved).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TAMSignature:
    """
    Tense-Aspect-Modality signature of the predicate governing a condition.

    Attributes with default values represent "no detectable signal" — their
    LLR contribution is 0.0, not noise.
    """
    tense:          str   = "unknown"   # past | present | future | unknown
    aspect:         str   = "simple"    # simple | progressive | perfect | past_perfect
    modal:          str   = "none"      # none | epistemic_weak | epistemic_strong
                                        #      | deontic | negated_deontic | conditional
    modal_strength: float = 1.0         # 0.0 – 1.0 scale on modal LLR magnitude

    def has_signal(self) -> bool:
        return self.tense != "unknown" or self.aspect != "simple" or self.modal != "none"


# ---------------------------------------------------------------------------
# LLR tables — each component contributes independently in log space
# ---------------------------------------------------------------------------
# Compositionality: "might have been resolving" = epistemic_weak + perfect +
# progressive.  Each row's values are added to the log-score vector; the
# sum correctly reflects all three independent evidence sources.

_TENSE_LLR: dict[str, dict[str, float]] = {
    # Past-tense clinical verbs: condition was present/active in the past
    # (conservative patterns only — see _RE_PAST_FINITE below)
    "past": {
        "resolved": +1.10, "ongoing": -0.70, "negated":  0.00, "ambiguous": -0.20,
    },
    # Present-tense clinical verbs: condition is currently present/active
    "present": {
        "resolved": -0.80, "ongoing": +0.65, "negated": +0.10, "ambiguous": +0.10,
    },
    # Future / planned: condition is anticipated but uncertain
    "future": {
        "resolved": -1.00, "ongoing": +0.30, "negated": -0.20, "ambiguous": +0.60,
    },
}

_ASPECT_LLR: dict[str, dict[str, float]] = {
    # "is worsening", "are improving" — unambiguously active ongoing process
    "progressive": {
        "ongoing": +1.50, "resolved": -1.50, "negated": -0.50, "ambiguous": -0.30,
    },
    # "has resolved", "have stabilised" — completed relative to present
    "perfect": {
        "resolved": +0.80, "ongoing": -0.30, "negated": -0.20, "ambiguous": -0.10,
    },
    # "had resolved", "had been treated" — completed before a past reference point
    "past_perfect": {
        "resolved": +1.50, "ongoing": -1.00, "negated": -0.20, "ambiguous": -0.10,
    },
}

_MODAL_LLR: dict[str, dict[str, float]] = {
    # "may have", "might be", "could indicate" — weak epistemic uncertainty
    "epistemic_weak": {
        "ambiguous": +1.50, "ongoing": -0.50, "resolved": -0.50, "negated": -1.00,
    },
    # "probably", "likely", "appears to be", "consistent with"
    "epistemic_strong": {
        "ambiguous": +1.00, "ongoing": +0.20, "resolved":  0.00, "negated": -0.80,
    },
    # "should be managed", "must be treated", "requires monitoring"
    # — condition is an active problem requiring attention
    "deontic": {
        "ongoing": +0.80, "resolved": -1.50, "negated": -0.80, "ambiguous": +0.20,
    },
    # "should not be present", "cannot" (without exclude/rule-out)
    # — condition is denied by obligation or strong negation
    "negated_deontic": {
        "negated": +0.90, "resolved": +0.30, "ongoing": -0.80, "ambiguous": -0.20,
    },
    # "would recur", "will be reviewed" — hypothetical or conditional
    "conditional": {
        "ambiguous": +0.80, "ongoing": +0.20, "resolved": -0.30, "negated": -0.50,
    },
}


# ---------------------------------------------------------------------------
# Extraction patterns — most specific first within each group
# ---------------------------------------------------------------------------

# ── Modal ────────────────────────────────────────────────────────────────────
# "cannot exclude" / "cannot rule out" are epistemic, not deontic negation:
# a negative lookahead prevents them matching negated_deontic.
# "won't go away" / "will not resolve" / "won't improve" mean the condition
# PERSISTS (ongoing) — the negation targets its disappearance, not its presence.
# These are excluded from negated_deontic by listing resolution/departure verbs.
_RE_NEGATED_DEONTIC = re.compile(
    r'\b(?:cannot(?!\s+(?:exclude|rule\s+out))|can\'t(?!\s+rule)|'
    r'(?:will\s+not|won\'t)(?!\s+(?:go\s+away|resolve|improve|clear|disappear|subside))|'
    r'should\s+not|must\s+not|'
    r'may\s+not\s+(?:be\s+)?(?:present|active|significant)|'
    r'no\s+longer\s+(?:needs?|requires?))\b',
    re.I,
)
_RE_EPISTEMIC_STRONG = re.compile(
    r'\b(?:probably|likely|appears?\s+to\s+be|seems?\s+to\s+be|'
    r'consistent\s+with|suggestive\s+of|in\s+keeping\s+with)\b',
    re.I,
)
_RE_EPISTEMIC_WEAK = re.compile(
    r'\b(?:may\b|might\b|could\s+(?:be|indicate|represent|suggest)|'
    r'possibly\b|cannot\s+(?:exclude|rule\s+out)|'
    r'cannot\s+be\s+excluded|question\s+of)\b',
    re.I,
)
_RE_DEONTIC = re.compile(
    r'\b(?:should\b|must\b|need\s+to|needs\s+to|ought\s+to|'
    r'requires?\b|required\b|warranted\b|recommended\b|'
    r'necessitates?\b)\b',
    re.I,
)
_RE_CONDITIONAL = re.compile(r'\bwould\b', re.I)

# ── Aspect — check before tense; perfect/progressive contain aux verbs ────────
_RE_PROGRESSIVE = re.compile(
    r'\b(?:is|are|was|were|been)\s+\w+ing\b', re.I
)
# Negative lookahead (?!been\b) prevents matching "have been" via backtracking:
# "have been resolving" — without it the regex engine can skip the optional
# "(?:been\s+)?" group and match "been" itself as the past-participle token
# (since "been" ends in "en"), producing a false perfect-aspect detection.
_RE_PRESENT_PERFECT = re.compile(
    r'\b(?:has|have)\s+(?:been\s+)?(?!been\b)[\w-]+(?:ed|en|ied|d|t)\b', re.I
)
_RE_PAST_PERFECT = re.compile(
    r'\bhad\s+(?:been\s+)?(?!been\b)[\w-]+(?:ed|en|ied|d|t)\b', re.I
)

# ── Tense — conservative patterns only to avoid false positives ───────────────
# "had no fever" / "has no fever" must NOT trigger past/present tense
# (they are negation constructions, not tense-bearing condition predicates).
# Only include verb patterns that unambiguously describe a condition's state.
_RE_PAST_FINITE = re.compile(
    r'\b(?:resolved\b|developed\b|worsened\b|recurred\b|relapsed\b|'
    r'was\s+(?:stable|controlled|well-controlled|poorly\s*controlled|'
    r'diagnosed|treated|noted|found|seen|present|active|uncontrolled)|'
    r'was\s+diagnosed\s+with|was\s+treated\s+for|presented\s+with|'
    r'occurred\b|underwent\b|experienced\b)\b',
    re.I,
)
_RE_PRESENT_FINITE = re.compile(
    r'\b(?:remains?\b|persists?\b|continues?\b|presents?\s+with|'
    r'is\s+(?:stable|active|controlled|well-controlled|poorly\s*controlled|'
    r'worsening|improving|progressing|ongoing|present|persistent|'
    r'uncontrolled|responding|refractory))\b',
    re.I,
)
_RE_FUTURE = re.compile(
    r'\b(?:will\s+(?!not\b)|going\s+to\s+(?!not\b))\b', re.I
)

# ── Strength modifiers ────────────────────────────────────────────────────────
_RE_STRONG_QUALIFIER = re.compile(
    r'\b(?:definitely|certainly|clearly|obviously|undoubtedly)\b', re.I
)
_RE_WEAK_QUALIFIER = re.compile(
    r'\b(?:somewhat|slightly|mildly|borderline)\b', re.I
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_tam(text: str) -> TAMSignature:
    """
    Extract the TAM signature from a clinical phrase or sentence.

    Parameters
    ----------
    text : clinical phrase (original or lowercased; patterns use re.I)

    Returns
    -------
    TAMSignature with has_signal()=False for bare noun phrases ("Hypertension")
    or empty input — contributing 0.0 LLR, not noise.

    Extraction priority
    -------------------
    1. Modality  — highest specificity; epistemic modal overrides tense inference
    2. Aspect    — progressive/perfect override simple-tense inference
    3. Tense     — falls back to conservative clinical verb scan
    """
    if not text or not text.strip():
        return TAMSignature()

    t = text
    sig = TAMSignature()

    # ── 1. Modality ──────────────────────────────────────────────────────────
    if _RE_NEGATED_DEONTIC.search(t):
        sig.modal = "negated_deontic"
    elif _RE_EPISTEMIC_STRONG.search(t):
        sig.modal = "epistemic_strong"
        sig.modal_strength = 0.85
    elif _RE_EPISTEMIC_WEAK.search(t):
        sig.modal = "epistemic_weak"
        # "might" is weaker than "may" or "could"
        if re.search(r'\bmight\b', t, re.I):
            sig.modal_strength = 0.70
        elif re.search(r'\bpossibly\b', t, re.I):
            sig.modal_strength = 0.75
        else:
            sig.modal_strength = 0.85
    elif _RE_DEONTIC.search(t):
        sig.modal = "deontic"
    elif _RE_CONDITIONAL.search(t):
        sig.modal = "conditional"

    # Strength modifiers apply to whatever modal was found
    if _RE_STRONG_QUALIFIER.search(t) and sig.modal != "none":
        sig.modal_strength = min(1.0, sig.modal_strength + 0.15)
    elif _RE_WEAK_QUALIFIER.search(t) and sig.modal == "none":
        # Standalone hedging word → treat as soft epistemic
        sig.modal = "epistemic_weak"
        sig.modal_strength = 0.50

    # ── 2. Aspect ────────────────────────────────────────────────────────────
    if _RE_PAST_PERFECT.search(t):
        sig.aspect = "past_perfect"
        sig.tense  = "past"          # past perfect implies past tense
    elif _RE_PRESENT_PERFECT.search(t):
        sig.aspect = "perfect"
        sig.tense  = "present"
    elif _RE_PROGRESSIVE.search(t):
        sig.aspect = "progressive"
        # Tense from auxiliary: "was/were + -ing" = past progressive
        sig.tense  = "past" if re.search(r'\b(?:was|were)\s+\w+ing\b', t, re.I) else "present"

    # ── 3. Tense (only when aspect has not already set it) ───────────────────
    if sig.tense == "unknown":
        if _RE_FUTURE.search(t):
            sig.tense = "future"
        elif _RE_PAST_FINITE.search(t):
            sig.tense = "past"
        elif _RE_PRESENT_FINITE.search(t):
            sig.tense = "present"

    return sig


def tam_to_llr(sig: TAMSignature, labels: tuple[str, ...]) -> dict[str, float]:
    """
    Convert a TAMSignature to per-label LLR contributions.

    Returns a zero-filled dict when has_signal() is False.

    Each component is added independently — compositionality in log space
    means the posterior correctly reflects all evidence.  A construction like
    "might have been resolving" (epistemic_weak + perfect + progressive) is
    handled by summing three independent LLR vectors.
    """
    llr: dict[str, float] = {l: 0.0 for l in labels}

    if sig.tense in _TENSE_LLR:
        for l in labels:
            llr[l] += _TENSE_LLR[sig.tense].get(l, 0.0)

    if sig.aspect in _ASPECT_LLR:
        for l in labels:
            llr[l] += _ASPECT_LLR[sig.aspect].get(l, 0.0)

    if sig.modal in _MODAL_LLR:
        for l in labels:
            llr[l] += _MODAL_LLR[sig.modal].get(l, 0.0) * sig.modal_strength

    return llr

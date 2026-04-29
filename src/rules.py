"""
Weighted cue definitions for clinical condition status classification.

Each entry is a (phrase, weight) tuple where weight ∈ (0, 1].
Higher weight = stronger / more reliable signal.

Design principles
-----------------
* Longer / more specific phrases appear first and carry higher weights.
  A four-word phrase like "no evidence of" is a much stronger negation signal
  than bare "no", so it gets weight 1.0 while bare "no" gets 0.6.
* Compound cues (e.g. "has no", "no active") encode common negation-scope
  patterns that single-word matching misses.
* Pseudo-negation patterns (PSEUDO_NEGATION_PATTERNS) are phrases that look
  like negations syntactically but are *not* denials of a condition.  They are
  detected before negation scoring and those spans are excluded.
"""

# ---------------------------------------------------------------------------
# NEGATION cues
# ---------------------------------------------------------------------------
# Ordered: most specific / highest-weight first.
NEGATION_CUES: list[tuple[str, float]] = [
    # --- multi-word strong negations ---
    ("patient denies any", 1.0),
    ("patient denies", 1.0),
    ("no evidence of", 1.0),
    ("no signs of", 1.0),
    ("no sign of", 1.0),
    ("no complaints of", 1.0),
    ("no report of", 1.0),
    ("no history of", 0.95),
    ("without evidence of", 1.0),
    ("without signs of", 1.0),
    ("negative for", 1.0),
    ("not present", 1.0),
    ("not found", 1.0),
    ("not detected", 1.0),
    ("not observed", 0.95),
    ("not reported", 0.9),
    ("not consistent with", 0.9),
    ("does not have", 1.0),
    ("did not have", 1.0),
    ("never had", 1.0),
    ("never diagnosed", 1.0),
    ("ruled out", 1.0),
    ("rules out", 1.0),
    ("free of", 0.9),
    ("clear of", 0.9),
    ("unremarkable for", 0.85),
    # --- compound scope cues ("verb + no/without") ---
    # These handle phrases like "patient has no fever" where bare "no" alone
    # would lose to the stronger "patient has" ongoing cue.
    ("has no", 0.95),
    ("have no", 0.95),
    ("had no", 0.90),
    ("shows no", 0.95),
    ("show no", 0.95),
    ("with no", 0.85),
    ("found no", 0.90),
    ("reveals no", 0.95),
    ("reveal no", 0.95),
    ("reports no", 0.90),
    ("without any", 0.85),
    ("no active", 0.90),
    # --- single-word negations (lower weight — easy false positives) ---
    ("denies", 1.0),
    ("absent", 0.85),
    ("none", 0.75),
    ("without", 0.70),
    ("not", 0.65),
    ("no", 0.60),
]

# ---------------------------------------------------------------------------
# RESOLVED cues
# ---------------------------------------------------------------------------
RESOLVED_CUES: list[tuple[str, float]] = [
    # --- strong multi-word historical phrases ---
    ("past medical history of", 1.0),
    ("past history of", 1.0),
    ("prior history of", 1.0),
    ("history of", 0.95),
    ("no longer has", 1.0),
    ("no longer present", 1.0),
    ("no longer", 0.95),
    ("status post", 1.0),
    ("s/p", 1.0),
    # --- recovery idioms often missed by keyword systems ---
    ("gotten over", 0.90),
    ("got over", 0.90),
    ("get over it", 0.85),
    ("completely over", 0.90),  # matches "got completely over" where words split "got over"
    ("fully over", 0.90),
    ("all better", 0.85),
    ("back to normal", 0.85),
    ("back to baseline", 0.90),
    ("returned to baseline", 0.90),
    ("feels better now", 0.80),
    ("has resolved", 1.0),
    ("have resolved", 1.0),
    ("completely resolved", 1.0),
    ("fully resolved", 1.0),
    ("resolved after", 1.0),
    ("in remission", 1.0),
    ("was treated for", 0.95),
    ("completed treatment for", 0.95),
    ("completed treatment", 0.90),
    ("after treatment", 0.85),
    ("following treatment", 0.85),
    ("post-treatment", 0.85),
    ("post-op", 0.85),
    ("postoperative", 0.85),
    ("previous episode of", 0.95),
    ("prior episode of", 0.95),
    ("recovered from", 0.95),
    ("recovery from", 0.90),
    ("in the past", 0.90),
    # --- single-word resolved signals ---
    ("resolved", 0.95),
    ("remission", 0.95),
    ("healed", 0.90),
    ("cleared", 0.85),
    ("recovered", 0.85),
    ("previous", 0.80),
    ("prior", 0.80),
    ("former", 0.80),
    ("formerly", 0.85),
    ("historically", 0.80),
    ("past", 0.70),
    ("discontinued", 0.75),
    ("discharged", 0.65),
]

# ---------------------------------------------------------------------------
# ONGOING cues
# ---------------------------------------------------------------------------
ONGOING_CUES: list[tuple[str, float]] = [
    # --- strong multi-word active phrases ---
    ("currently active", 1.0),
    ("currently experiencing", 1.0),
    ("currently has", 0.95),
    ("continues to have", 0.95),
    ("continues to experience", 0.95),
    ("presents with", 0.95),
    ("presenting with", 0.95),
    ("complains of", 0.95),
    ("complaints of", 0.95),
    ("still has", 0.95),
    ("still present", 0.95),
    ("at this time", 0.85),
    ("at present", 0.85),
    ("poorly controlled", 0.95),
    ("uncontrolled", 0.95),
    ("well-controlled", 0.90),
    # --- single-word strong ongoing ---
    ("currently", 0.90),
    ("ongoing", 1.0),
    ("persistent", 1.0),
    ("persists", 1.0),
    ("persisting", 1.0),
    ("worsening", 0.95),
    ("deteriorating", 0.95),
    ("exacerbation", 0.95),
    ("flaring", 0.95),
    ("flare", 0.90),
    ("progressing", 0.90),
    ("progressive", 0.90),
    ("chronic", 0.85),
    ("chronically", 0.85),
    # --- moderate ongoing ---
    ("actively", 0.85),
    ("active", 0.80),
    ("stable", 0.80),
    ("controlled", 0.80),
    ("improving", 0.75),
    ("better", 0.70),
    ("worse", 0.80),
    ("continuing", 0.80),
    ("continued", 0.75),
    ("recurrent", 0.85),
    ("recurring", 0.85),
    # --- weak / generic (high false-positive risk → low weight) ---
    ("reports", 0.65),
    ("patient has", 0.70),
    ("has", 0.50),
    ("with", 0.40),
]

# ---------------------------------------------------------------------------
# AMBIGUOUS cues
# ---------------------------------------------------------------------------
AMBIGUOUS_CUES: list[tuple[str, float]] = [
    # --- high-confidence uncertainty phrases ---
    ("cannot rule out", 1.0),
    ("can not rule out", 1.0),
    ("unable to rule out", 1.0),
    ("concern for", 1.0),
    ("concerning for", 1.0),
    ("question of", 1.0),
    ("may have", 1.0),
    ("might have", 0.95),
    ("could be", 0.90),
    ("suspicious for", 0.95),
    ("differential includes", 0.90),
    ("differential diagnosis", 0.85),
    ("cannot exclude", 0.90),
    ("not excluded", 0.85),
    # --- single-word uncertainty ---
    ("rule out", 0.95),
    ("r/o", 0.95),
    ("possible", 0.95),
    ("possibly", 0.95),
    ("probable", 0.90),
    ("probably", 0.90),
    ("suspected", 0.95),
    ("suspect", 0.85),
    ("likely", 0.75),
    ("unlikely", 0.70),
    ("uncertain", 0.90),
    ("unclear", 0.85),
    ("indeterminate", 0.90),
    ("equivocal", 0.90),
    ("query", 0.80),
    ("consider", 0.70),
    ("may be", 0.85),
    # --- first-person hedging (common in patient-reported text) ---
    ("i think", 0.65),
    ("i believe", 0.60),
    ("i feel like", 0.60),
    ("not sure if", 0.75),
    ("not certain", 0.75),
]

# ---------------------------------------------------------------------------
# PSEUDO-NEGATION patterns (regex strings)
# ---------------------------------------------------------------------------
# These patterns contain negation words ("no", "not", "without") but do NOT
# deny the existence of a condition.  They must be detected and their spans
# excluded before negation scoring, otherwise "no longer" fires as "no"
# (negated) instead of the intended "resolved" signal.
PSEUDO_NEGATION_PATTERNS: list[str] = [
    r"\bnot only\b",            # additive ("not only X but also Y")
    r"\bnot just\b",            # additive
    r"\bnot merely\b",          # additive
    r"\bnot always\b",          # frequency, not denial
    r"\bnot necessarily\b",     # qualifier
    r"\bnot certain\b",         # uncertainty → ambiguous, not negated
    r"\bnot sure\b",            # uncertainty
    r"\bnot clear\b",           # uncertainty
    r"\bnot fully\b",           # partial (still somewhat present)
    r"\bnot completely\b",      # partial
    r"\bnot entirely\b",        # partial
    r"\bnot responding\b",      # ongoing (not responding to treatment)
    r"\bnot improving\b",       # ongoing (condition still present)
    r"\bnot well controlled\b", # ongoing
    r"\bno apparent\b",         # ambiguous (no apparent cause → still present)
    r"\bno change\b",           # ongoing (unchanged)
    r"\bno improvement\b",      # ongoing (condition persisting)
    r"\bno relief\b",           # ongoing
]

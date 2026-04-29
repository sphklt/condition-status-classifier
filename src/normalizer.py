"""
Clinical abbreviation normalizer.

Real clinical notes use heavy shorthand. Expanding abbreviations before
classification dramatically reduces missed cues — e.g. "h/o diabetes" fails
every rule in the original system because no cue list has "h/o".

Abbreviations are matched with word boundaries (\\b) so "sp" inside a word
is not expanded, only standalone tokens.
"""

import re

# Order matters: longer / more specific patterns must come first so they are
# matched and replaced before their sub-strings are attempted.
_ABBREVIATIONS: list[tuple[str, str]] = [
    # --- history / context ---
    (r"\bpmhx\b", "past medical history of"),
    (r"\bpmh\b", "past medical history"),
    (r"\bh/o\b", "history of"),
    (r"\bhx\b", "history"),
    # --- status / procedure ---
    (r"\bs/p\b", "status post"),
    # --- presenting complaint ---
    (r"\bc/o\b", "complains of"),
    # --- diagnosis ---
    (r"\bdx\b", "diagnosed with"),
    # --- rule out ---
    (r"\br/o\b", "rule out"),
    (r"\bro\b", "rule out"),
    # --- without ---
    (r"\bw/o\b", "without"),
    # --- lab / test results ---
    (r"(?<!\w)-ve(?!\w)", "negative for"),
    (r"\b\+ve\b", "positive for"),
    (r"\bneg\b", "negative for"),
    # --- common disease abbreviations (expand so ongoing cues fire) ---
    (r"\bhtn\b", "hypertension"),
    (r"\bdm2\b", "type 2 diabetes mellitus"),
    (r"\bdm1\b", "type 1 diabetes mellitus"),
    (r"\bdm\b", "diabetes mellitus"),
    (r"\bcad\b", "coronary artery disease"),
    (r"\bchf\b", "congestive heart failure"),
    (r"\bcopd\b", "chronic obstructive pulmonary disease"),
    (r"\bafib\b", "atrial fibrillation"),
    (r"\buti\b", "urinary tract infection"),
    (r"\bpe\b", "pulmonary embolism"),
    (r"\bdvt\b", "deep vein thrombosis"),
    (r"\bmi\b", "myocardial infarction"),
    (r"\bcva\b", "cerebrovascular accident"),
    (r"\bra\b", "rheumatoid arthritis"),
    (r"\bsle\b", "systemic lupus erythematosus"),
    # --- patient shorthand ---
    (r"\bpt\b", "patient"),
    (r"\bpts\b", "patients"),
    # --- resolved shorthand ---
    (r"\bresol\b", "resolved"),
    # --- "no." used as abbreviation for "number" — NOT negation ---
    (r"\bno\.\s", "number "),
]

# Compile once for performance
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _ABBREVIATIONS
]


def normalize(text: str) -> tuple[str, list[str]]:
    """
    Expand clinical abbreviations in *text*.

    Returns (expanded_text, list_of_expansions_applied) so callers can surface
    which abbreviations were found (useful for explainability in the UI).
    """
    result = text
    expansions: list[str] = []
    for pattern, replacement in _COMPILED:
        new_result, n = pattern.subn(replacement, result)
        if n > 0:
            original_abbrev = pattern.pattern.strip(r"\b")
            expansions.append(f"{original_abbrev} → {replacement}")
            result = new_result
    return result, expansions

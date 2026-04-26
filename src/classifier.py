from src.rules import (
    NEGATION_CUES,
    RESOLVED_CUES,
    ONGOING_CUES,
    AMBIGUOUS_CUES,
)


def find_matching_cue(text: str, cues: list[str]) -> str | None:
    """
    Returns the first cue found in the text.
    If no cue is found, returns None.
    """
    text_lower = text.lower()

    for cue in cues:
        if cue in text_lower:
            return cue

    return None


def classify_condition_status(text: str) -> dict:
    """
    Classifies a clinical phrase into one of:
    - negated
    - ambiguous
    - resolved
    - ongoing

    Priority matters:
    1. Negation
    2. Ambiguity
    3. Resolved
    4. Ongoing
    5. Default ongoing

    Why default ongoing?
    If a phrase simply says "asthma" or "patient has asthma",
    it is usually safer to treat it as active unless there is
    evidence saying it is resolved, negated, or uncertain.
    """

    if not text or not text.strip():
        return {
            "status": "ambiguous",
            "cue": None,
            "reason": "Empty or missing text",
        }

    negation_cue = find_matching_cue(text, NEGATION_CUES)
    if negation_cue:
        return {
            "status": "negated",
            "cue": negation_cue,
            "reason": f"Negation cue found: '{negation_cue}'",
        }

    ambiguous_cue = find_matching_cue(text, AMBIGUOUS_CUES)
    if ambiguous_cue:
        return {
            "status": "ambiguous",
            "cue": ambiguous_cue,
            "reason": f"Uncertainty cue found: '{ambiguous_cue}'",
        }

    resolved_cue = find_matching_cue(text, RESOLVED_CUES)
    if resolved_cue:
        return {
            "status": "resolved",
            "cue": resolved_cue,
            "reason": f"Resolved or historical cue found: '{resolved_cue}'",
        }

    ongoing_cue = find_matching_cue(text, ONGOING_CUES)
    if ongoing_cue:
        return {
            "status": "ongoing",
            "cue": ongoing_cue,
            "reason": f"Ongoing/active cue found: '{ongoing_cue}'",
        }

    return {
        "status": "ongoing",
        "cue": None,
        "reason": "No closing, negation, or uncertainty cue found; defaulting to ongoing",
    }
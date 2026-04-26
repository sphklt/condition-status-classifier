"""
Rule definitions for clinical condition status classification.

The goal is not to be medically perfect.
The goal is to create an explainable baseline that can classify
short clinical phrases into:
- ongoing
- resolved
- negated
- ambiguous
"""

NEGATION_CUES = [
    "denies",
    "no evidence of",
    "negative for",
    "without",
    "no signs of",
    "not present",
]

RESOLVED_CUES = [
    "resolved",
    "no longer",
    "history of",
    "past history of",
    "previous",
    "prior",
    "status post",
    "s/p",
]

ONGOING_CUES = [
    "has",
    "currently",
    "active",
    "persistent",
    "ongoing",
    "worsening",
    "stable",
    "controlled",
    "improving",
    "better",
]

AMBIGUOUS_CUES = [
    "possible",
    "rule out",
    "r/o",
    "suspected",
    "may have",
    "concern for",
    "question of",
]
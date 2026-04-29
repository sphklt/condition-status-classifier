"""
Clinical note section detector.

Splits free-text clinical notes into labeled sections using regex header
patterns. Each section carries a status_prior — a strong baseline signal the
classifier can use when its own cue-matching confidence is low.

Why this matters
----------------
A condition mentioned in "Past Medical History" is almost certainly resolved.
One in "Chief Complaint" is almost certainly ongoing. The current phrase
classifier ignores this entirely; the pipeline uses it as a tiebreaker.
"""

import re
from dataclasses import dataclass


@dataclass
class NoteSection:
    name: str               # canonical section name
    header: str             # the matched header text
    text: str               # section body
    status_prior: str | None  # "ongoing", "resolved", "negated", or None


# (canonical_name, [regex sub-patterns], status_prior)
# Ordered from most specific to least specific within each group.
_SECTION_DEFS: list[tuple[str, list[str], str | None]] = [
    ("past_medical_history", [
        r"past\s+medical\s+history",
        r"pmhx?",
        r"medical\s+history",
        r"past\s+history",
    ], "resolved"),

    ("past_surgical_history", [
        r"past\s+surgical\s+history",
        r"psh",
        r"surgical\s+history",
        r"operative\s+history",
    ], "resolved"),

    ("family_history", [
        r"family\s+history",
        r"fhx?",
    ], None),  # family conditions — not the patient's own status

    ("chief_complaint", [
        r"chief\s+complaint",
        r"presenting\s+complaint",
        r"reason\s+for\s+(?:visit|referral)",
    ], "ongoing"),

    ("history_of_present_illness", [
        r"history\s+of\s+present\s+illness",
        r"hpi",
        r"present(?:ing)?\s+illness",
    ], "ongoing"),

    ("review_of_systems", [
        r"review\s+of\s+systems",
        r"ros\b",
    ], None),  # mixed — classifier decides

    ("assessment", [
        r"assessment\s+and\s+plan",
        r"assessment",
        r"impression",
        r"a\s*/\s*p",        # A/P
        r"working\s+diagnosis",
    ], "ongoing"),

    ("plan", [
        r"\bplan\b",
        r"treatment\s+plan",
        r"management",
    ], None),

    ("medications", [
        r"current\s+medications?",
        r"\bmedications?\b",
        r"\bmeds\b",
    ], "ongoing"),

    ("allergies", [
        r"(?:known\s+)?(?:drug\s+)?allergies",
        r"\bnkda\b",
    ], "negated"),

    ("physical_examination", [
        r"physical\s+exam(?:ination)?",
        r"exam(?:ination)?\s+findings",
    ], None),

    ("social_history", [
        r"social\s+history",
        r"shx?",
    ], None),
]

# Pre-compile: one regex per section that matches a header line.
# Headers appear at the start of a line (possibly with leading whitespace)
# and are optionally followed by a colon.
_COMPILED: list[tuple[str, re.Pattern, str | None]] = []
for _name, _patterns, _prior in _SECTION_DEFS:
    _combined = "|".join(rf"(?:{p})" for p in _patterns)
    _re = re.compile(
        rf"(?:(?<=\n)|^)\s*(?:{_combined})\s*:?[ \t]*(?=\n|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    _COMPILED.append((_name, _re, _prior))


def detect_sections(text: str) -> list[NoteSection]:
    """
    Split *text* into labeled NoteSection objects.

    Any content before the first detected header is returned as section
    "unknown" with no status prior.  If no headers are found at all, the
    entire note is returned as a single "unknown" section.
    """
    hits: list[tuple[int, int, str, str | None]] = []
    for name, pattern, prior in _COMPILED:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.end(), name, prior, m.group().strip()))

    if not hits:
        return [NoteSection("unknown", "", text.strip(), None)]

    hits.sort(key=lambda x: x[0])

    sections: list[NoteSection] = []

    # Preamble before the first header
    if hits[0][0] > 0:
        preamble = text[: hits[0][0]].strip()
        if preamble:
            sections.append(NoteSection("unknown", "", preamble, None))

    for i, (start, end, name, prior, header) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[end:next_start].strip()
        sections.append(NoteSection(name, header, body, prior))

    return sections

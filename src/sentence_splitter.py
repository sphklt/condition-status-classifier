"""
Clinical sentence boundary detector.

Problem it solves
-----------------
The pipeline's original fixed-char context window leaked signals from
adjacent sentences into entity classification. Example:

    "No fever.  Patient has diabetes."
     ^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
    sentence 1    sentence 2

When classifying "diabetes", the 120-char window included "No fever",
causing the negation cue "No" to bleed into the diabetes score.

This module splits text into sentences — respecting clinical abbreviations
so "Dr. Smith" and "3.5 mg" don't create false boundaries — and returns
each sentence with its character offsets so the pipeline can map NER
entity positions back to their containing sentence.

No external dependencies required.
"""

import re
from dataclasses import dataclass


@dataclass
class Sentence:
    text: str
    start: int  # char offset in the original text (inclusive)
    end: int    # char offset (exclusive)


# ---------------------------------------------------------------------------
# Abbreviation protection
# ---------------------------------------------------------------------------
# Words that end with a period but are NOT sentence boundaries.
# Listed in rough frequency order; longer entries come first so the
# replacement loop matches them before their sub-strings.
_ABBREVS: frozenset[str] = frozenset({
    # Titles — classic false-boundary source: "Dr. Smith", "Prof. Jones"
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    # Latin / general — appear mid-sentence: "e.g. asthma", "i.e. pain"
    "vs", "etc", "eg", "ie", "al", "et", "cf", "approx",
    "est", "avg", "dept", "fig", "vol", "no", "inc",
    # Dose frequency — appear mid-phrase: "500 mg b.i.d. dosing schedule"
    # (units themselves are NOT included — "mg. She" is always a real boundary)
    "b.i.d", "t.i.d", "q.i.d", "bid", "tid", "qid",
    "q.d", "q.h.s",
})

# A null byte is safe inside medical text and makes a clean placeholder.
_GUARD = "\x00"


def _protect_periods(text: str) -> str:
    """
    Replace periods that are NOT sentence boundaries with _GUARD so the
    boundary regex ignores them.
    """
    result = text

    # 1. Decimal numbers: "3.5" → "3\x005"
    result = re.sub(r"(\d+)\.(\d)", lambda m: m.group(1) + _GUARD + m.group(2), result)

    # 2. Known abbreviations (longest first to avoid partial matches)
    for abbrev in sorted(_ABBREVS, key=len, reverse=True):
        result = re.sub(
            r"\b" + re.escape(abbrev) + r"\.",
            abbrev + _GUARD,
            result,
            flags=re.IGNORECASE,
        )

    # 3. Single capital-letter initials: "J." → "J\x00"
    result = re.sub(r"\b([A-Z])\.", r"\1" + _GUARD, result)

    return result


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# A sentence boundary is one-or-more .!? followed by whitespace,
# where the next non-space char is a capital letter, digit, list marker,
# or end of string.
_BOUNDARY = re.compile(r"[.!?]+\s+")
_SENTENCE_START = re.compile(r"[A-Z\d\-•*]")


def split_sentences(text: str) -> list[Sentence]:
    """
    Split *text* into Sentence objects, each with its char offset in *text*.

    Positions are relative to the original text (before abbreviation
    protection), so they are directly usable with NER entity offsets that
    were also computed on the same text.
    """
    if not text or not text.strip():
        return []

    protected = _protect_periods(text)

    # Collect candidate split points from the protected text.
    split_starts: list[int] = [0]
    for m in _BOUNDARY.finditer(protected):
        next_char = protected[m.end() : m.end() + 1]
        if next_char and _SENTENCE_START.match(next_char):
            split_starts.append(m.end())

    split_starts.append(len(protected))

    sentences: list[Sentence] = []
    for i in range(len(split_starts) - 1):
        start = split_starts[i]
        end = split_starts[i + 1]

        # Use the ORIGINAL text for the sentence body (not the protected version)
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue

        leading_ws = len(raw) - len(raw.lstrip())
        sentences.append(Sentence(stripped, start + leading_ws, end))

    return sentences


def find_sentence_context(
    sentences: list[Sentence],
    entity_start: int,
    entity_end: int,
) -> str:
    """
    Return the sentence that contains the entity span [entity_start, entity_end).

    If the entity straddles a boundary (rare), returns all sentences that
    overlap with the span joined by a space.

    Falls back to an empty string if sentences is empty.
    """
    if not sentences:
        return ""

    containing: list[str] = []
    for sent in sentences:
        # Entity sits inside this sentence
        if sent.start <= entity_start < sent.end:
            containing.append(sent.text)
            break
        # Entity straddles this sentence boundary
        if sent.start < entity_end and sent.end > entity_start:
            containing.append(sent.text)

    return " ".join(containing) if containing else sentences[0].text

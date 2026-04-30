"""
Optional dependency-parsing layer for negation scope and list negation.

Why this exists
---------------
Regex-based negation matching fires on ANY negation word in the sentence,
regardless of which noun it actually governs in the parse tree:

    "Patient has no fever but has hypertension."
     negation fires on "no" → hypertension incorrectly scored as negated

    "Denies fever, chills, or chest pain."
     only "fever" (first object) is correctly attributed; list negation is
     handled coincidentally, not by design.

Dependency parsing fixes both:
  1. Negation scope  — trace the dep tree from "no"/"not"/"denies" to its
                       governed subtree; only flag an entity as negated if it
                       falls inside that subtree.
  2. List negation   — detect "deny/negative" verbs and enumerate ALL direct
                       objects (including conjuncts), returning every item as
                       an explicitly negated term.

Falls back gracefully
---------------------
If en_core_web_sm is not installed, every public function returns None (scope
checks) or [] (list extraction). Callers are expected to fall back to the
existing regex classifier in that case.
"""

import re

_nlp = None
_AVAILABLE: bool | None = None   # None = not yet tried


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load() -> bool:
    global _nlp, _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        _AVAILABLE = True
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


def dep_parser_available() -> bool:
    """Return True if en_core_web_sm is installed and loaded."""
    return _load()


# ---------------------------------------------------------------------------
# Negation scope
# ---------------------------------------------------------------------------

def check_negation_scope(sentence: str, entity_text: str) -> bool | None:
    """
    Return whether *entity_text* is inside the scope of a negation in *sentence*.

    Returns
    -------
    True  — a negation token governs the entity's subtree
    False — no negation found, or negation does not govern the entity
    None  — parser not available; caller should fall back to regex
    """
    if not _load():
        return None

    doc = _nlp(sentence)
    entity_lower = entity_text.lower()
    text_lower = sentence.lower()

    # Locate entity character span in the sentence
    pos = text_lower.find(entity_lower)
    if pos == -1:
        return None
    entity_char_end = pos + len(entity_lower)

    # Tokens belonging to the entity
    ent_tokens = [
        t for t in doc
        if t.idx >= pos and t.idx + len(t.text) <= entity_char_end
    ]
    if not ent_tokens:
        return None

    # Negation tokens: explicit dep_=neg, or negative determiners ("no", "not")
    neg_tokens = [
        t for t in doc
        if t.dep_ == "neg"
        or (t.dep_ == "det" and t.lemma_.lower() in ("no",))
        or t.lemma_.lower() in ("deny", "negative", "absent", "negate")
        and t.pos_ == "VERB"
    ]

    if not neg_tokens:
        return False

    # Check whether any entity token falls inside the subtree of the negation's head
    for neg_tok in neg_tokens:
        governed = set(neg_tok.head.subtree)
        for ent_tok in ent_tokens:
            if ent_tok in governed:
                return True

    return False


# ---------------------------------------------------------------------------
# List negation
# ---------------------------------------------------------------------------

_DENY_LEMMAS = frozenset({
    "deny", "negative", "absent", "negate", "lack", "exclude", "rule",
})


def extract_list_negated(sentence: str) -> list[str]:
    """
    Detect list-negation patterns like "Denies fever, chills, or chest pain."
    and return every negated item as a string.

    Returns [] if the parser is unavailable or no list negation is found.
    """
    if not _load():
        return []

    doc = _nlp(sentence)
    negated: list[str] = []

    for token in doc:
        if token.lemma_.lower() not in _DENY_LEMMAS:
            continue
        if token.pos_ not in ("VERB", "ADJ"):
            continue

        # Collect all direct objects and their conjuncts
        dobjs = [c for c in token.children if c.dep_ in ("dobj", "attr", "nsubj")]
        for dobj in dobjs:
            # The dobj itself
            negated.append(_span_text(dobj))
            # Any conjuncts of the dobj (e.g. "fever, chills, chest pain")
            for conj in dobj.conjuncts:
                negated.append(_span_text(conj))

    return [n for n in negated if n]


def _span_text(token) -> str:
    """Return the full noun-phrase text rooted at *token*."""
    # Collect compound/amod children that form part of the NP
    subtree_tokens = sorted(
        [t for t in token.subtree
         if t.dep_ in ("compound", "amod", "nmod", "poss") or t == token],
        key=lambda t: t.i,
    )
    if not subtree_tokens:
        return token.text
    return " ".join(t.text for t in subtree_tokens)


# ---------------------------------------------------------------------------
# Modifier-scope check (temporal / historical)
# ---------------------------------------------------------------------------

_TEMPORAL_LEMMAS = frozenset({
    "previously", "formerly", "historically", "prior", "past",
    "ago", "recently", "currently", "now", "today",
})


def temporal_modifies_entity(sentence: str, entity_text: str) -> bool | None:
    """
    Return whether a temporal modifier in *sentence* attaches to *entity_text*
    or to something else (e.g. a subordinate clause).

    Specifically useful for:
        "Atrial fibrillation, previously in sinus rhythm."
    where "previously" modifies "sinus rhythm", NOT "atrial fibrillation".

    Returns
    -------
    True  — temporal modifier is within the entity's governing subtree
    False — temporal modifier attaches elsewhere
    None  — parser unavailable or no temporal modifier found
    """
    if not _load():
        return None

    doc = _nlp(sentence)
    entity_lower = entity_text.lower()
    pos = sentence.lower().find(entity_lower)
    if pos == -1:
        return None
    entity_char_end = pos + len(entity_lower)

    ent_tokens = [
        t for t in doc
        if t.idx >= pos and t.idx + len(t.text) <= entity_char_end
    ]
    if not ent_tokens:
        return None

    temp_tokens = [
        t for t in doc
        if t.lemma_.lower() in _TEMPORAL_LEMMAS
    ]
    if not temp_tokens:
        return None

    # Find the entity's syntactic head (token whose head is outside the span)
    ent_set = set(t.i for t in ent_tokens)
    ent_head = ent_tokens[0]
    for t in ent_tokens:
        if t.head.i not in ent_set:
            ent_head = t
            break

    ent_subtree = set(ent_head.subtree)

    for temp_tok in temp_tokens:
        if temp_tok in ent_subtree:
            return True

    return False

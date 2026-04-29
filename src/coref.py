"""
Simple pronoun-to-entity coreference for the note pipeline.

When a sentence in a clinical section contains no NER entity but includes a
pronoun (it, this, they…) and a confident status signal, the status is
attributed to the most recently classified entity in the same section.

This handles common clinical narrative patterns such as:
    "The patient had a cough. It resolved."
    "She developed pneumonia. The condition was treated and cleared."

No dependency parsing required — heuristic: pronoun presence + high-confidence
classifier result on the pronoun sentence, compared to the entity's sentence.
"""

import re

_PRONOUN_RE = re.compile(
    r"\b(it|this|they|the condition|the infection|the disease|"
    r"the illness|the disorder|the symptom|the problem)\b",
    re.IGNORECASE,
)

# Minimum confidence on the pronoun sentence to trigger an update.
_CONFIDENCE_THRESHOLD = 0.70

# Coref only fires when the existing classification is weak.
# If the entity's own sentence already gave a strong signal (>= this value),
# the entity is already well-classified and coreference should not override it.
_MAX_EXISTING_CONFIDENCE = 0.65


def has_pronoun(text: str) -> bool:
    """Return True if *text* contains a coreference-eligible pronoun."""
    return bool(_PRONOUN_RE.search(text))


def apply_pronoun_coref(
    entity_positions: list[tuple[int, int]],
    results: list,
    sentences: list,
    classify_fn,
) -> int:
    """
    Scan sentences that contain no NER entity; if one has a pronoun and a
    confident status signal, attribute that status to the most recently
    classified entity in the same section (by text position).

    Parameters
    ----------
    entity_positions : list of (start, end)
        Character offsets of each entity, same order as *results*.
    results : list of ConditionResult
        Classified entities for this section — modified **in-place**.
    sentences : list of Sentence
        All sentences in the section (from split_sentences).
    classify_fn : callable
        classify_condition_status — injected to avoid circular imports.

    Returns
    -------
    int
        Number of entities whose classification was updated.
    """
    if not results or not sentences:
        return 0

    # Build: sentence_index → list of result indices whose entity falls there
    entity_sentence_map: dict[int, list[int]] = {}
    for r_idx, (e_start, _) in enumerate(entity_positions):
        for s_idx, sent in enumerate(sentences):
            if sent.start <= e_start < sent.end:
                entity_sentence_map.setdefault(s_idx, []).append(r_idx)
                break

    sent_has_entity: set[int] = set(entity_sentence_map.keys())
    updates = 0

    for s_idx, sentence in enumerate(sentences):
        if s_idx in sent_has_entity:
            continue
        if not has_pronoun(sentence.text):
            continue

        clf = classify_fn(sentence.text)
        if clf["confidence"] < _CONFIDENCE_THRESHOLD:
            continue

        # Find the most recent entity result whose sentence is BEFORE this one
        referent_idx = None
        for r_idx, (e_start, _) in enumerate(entity_positions):
            for cand_idx, cand_sent in enumerate(sentences):
                if cand_sent.start <= e_start < cand_sent.end:
                    if cand_idx < s_idx:
                        referent_idx = r_idx  # keep the latest one before s_idx
                    break

        if referent_idx is None:
            continue

        existing = results[referent_idx]
        new_conf = round(clf["confidence"] * 0.92, 3)  # discount: ignoring earlier context

        # Update only when:
        #  (a) status would change
        #  (b) existing classification was weak (entity's own sentence uninformative)
        #  (c) pronoun sentence is at least as confident as the existing result
        if (
            clf["status"] != existing.status
            and existing.confidence < _MAX_EXISTING_CONFIDENCE
            and new_conf >= existing.confidence
        ):
            results[referent_idx].status = clf["status"]
            results[referent_idx].confidence = new_conf
            results[referent_idx].reason = (
                f"[Coref: '{sentence.text.strip()[:50]}'] {clf['reason']}"
            )
            updates += 1

    return updates

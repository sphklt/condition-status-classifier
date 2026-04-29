"""
Full clinical note processing pipeline.

Transforms a raw clinical note into a structured list of
{condition → status} results by chaining:

  1. Section detection   — identify PMH, HPI, Assessment, etc.
  2. NER                 — extract condition/symptom spans per section
  3. Sentence splitting  — split section text into sentences with char offsets
  4. Context extraction  — sentence containing the entity (not a fixed char window)
  5. Classification      — run the existing phrase classifier on that sentence
  6. Section prior       — override low-confidence results using section context

Step 3/4 replaced the original fixed 120-char window. The old approach leaked
signals across sentence boundaries — e.g. "No fever. Patient has diabetes."
would bleed the negation cue "No" into the diabetes context. The sentence-aware
window eliminates this by classifying each entity within its own sentence only.
"""

import re
from dataclasses import dataclass, field

from src.section_detector import detect_sections, NoteSection
from src.ner import extract_entities, active_ner_method
from src.classifier import classify_condition_status
from src.normalizer import normalize
from src.sentence_splitter import split_sentences, find_sentence_context
from src.coref import apply_pronoun_coref

# When classifier confidence is below this threshold, the section's
# status prior (if any) overrides the classification.
_PRIOR_OVERRIDE_THRESHOLD = 0.55

# Sections whose conditions are not the patient's own — skip entirely.
_SKIP_SECTIONS = {"family_history"}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    condition: str          # entity text as found in the note
    status: str             # ongoing | resolved | negated | ambiguous
    confidence: float
    section: str            # canonical section name
    context: str            # text window that was classified
    reason: str             # human-readable explanation
    overridden_by_prior: bool = False  # True when section prior took over


@dataclass
class PipelineResult:
    conditions: list[ConditionResult] = field(default_factory=list)
    sections_found: list[str] = field(default_factory=list)
    ner_method: str = "vocabulary"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentence_context(
    sentences: list,
    entity_start: int,
    entity_end: int,
    fallback_text: str,
) -> str:
    """
    Return the sentence that contains the entity.
    Falls back to the full section text if sentence splitting produced nothing.
    """
    ctx = find_sentence_context(sentences, entity_start, entity_end)
    return ctx if ctx else fallback_text.strip()


def _deduplicate(results: list[ConditionResult]) -> list[ConditionResult]:
    """
    Keep only the first occurrence of each condition (by normalised text).
    Later sections may mention conditions already classified in higher-priority
    sections (e.g. PMH condition also appears in Plan).
    """
    seen: set[str] = set()
    out: list[ConditionResult] = []
    for r in results:
        key = r.condition.lower().strip()
        # Strip simple plurals for dedup
        key = re.sub(r"s$", "", key)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_note(note_text: str) -> PipelineResult:
    """
    Process a raw clinical note and return structured condition → status results.

    Parameters
    ----------
    note_text : str
        Free-text clinical note (discharge summary, clinic letter, SOAP note, etc.)

    Returns
    -------
    PipelineResult
        .conditions      — list of ConditionResult (one per detected entity)
        .sections_found  — canonical section names detected in the note
        .ner_method      — "scispacy" or "vocabulary"
        .warnings        — any non-fatal issues encountered
    """
    result = PipelineResult(ner_method=active_ner_method())

    if not note_text or not note_text.strip():
        result.warnings.append("Empty note text provided.")
        return result

    # ── Step 1: section detection ─────────────────────────────────────────
    sections = detect_sections(note_text)
    result.sections_found = [s.name for s in sections if s.name != "unknown"]

    raw_results: list[ConditionResult] = []

    for section in sections:
        if section.name in _SKIP_SECTIONS:
            continue
        if not section.text.strip():
            continue

        # ── Step 2: normalise + NER ───────────────────────────────────────
        normalized, _ = normalize(section.text)
        entities = extract_entities(normalized)

        if not entities and section.name != "unknown":
            result.warnings.append(
                f"No entities found in section '{section.name}'."
            )

        # ── Step 3: split section into sentences (done once per section) ──
        sentences = split_sentences(normalized)

        # ── Steps 4 + 5: sentence context + classification ────────────────
        section_results: list[ConditionResult] = []
        entity_positions: list[tuple[int, int]] = []

        for entity in entities:
            context = _sentence_context(sentences, entity.start, entity.end, normalized)
            clf = classify_condition_status(context)

            overridden = False

            # ── Step 5: section prior override ────────────────────────────
            if (
                section.status_prior is not None
                and clf["confidence"] < _PRIOR_OVERRIDE_THRESHOLD
            ):
                clf["status"] = section.status_prior
                clf["reason"] = (
                    f"Low-confidence classifier result overridden by "
                    f"section prior '{section.name}' → {section.status_prior}."
                )
                overridden = True

            entity_positions.append((entity.start, entity.end))
            section_results.append(ConditionResult(
                condition=entity.text,
                status=clf["status"],
                confidence=clf["confidence"],
                section=section.name,
                context=context,
                reason=clf["reason"],
                overridden_by_prior=overridden,
            ))

        # ── Step 6: pronoun coreference within this section ───────────────
        # Sentences with no entity (e.g. "It resolved.") may update the
        # most recently classified entity if a pronoun + confident signal
        # is found.
        apply_pronoun_coref(entity_positions, section_results, sentences, classify_condition_status)

        raw_results.extend(section_results)

    result.conditions = _deduplicate(raw_results)
    return result


def format_results(pipeline_result: PipelineResult) -> str:
    """
    Return a readable plain-text summary of pipeline results.
    Useful for CLI output and quick debugging.
    """
    lines: list[str] = []
    lines.append(f"NER method : {pipeline_result.ner_method}")
    lines.append(f"Sections   : {', '.join(pipeline_result.sections_found) or 'none detected'}")
    lines.append("")

    if not pipeline_result.conditions:
        lines.append("No conditions detected.")
        return "\n".join(lines)

    header = f"{'CONDITION':<35} {'STATUS':<12} {'CONF':>6}  {'SECTION'}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in pipeline_result.conditions:
        flag = " *" if r.overridden_by_prior else ""
        lines.append(
            f"{r.condition:<35} {r.status:<12} {r.confidence:>6.0%}  {r.section}{flag}"
        )

    if any(r.overridden_by_prior for r in pipeline_result.conditions):
        lines.append("\n* overridden by section prior (low classifier confidence)")

    if pipeline_result.warnings:
        lines.append("")
        for w in pipeline_result.warnings:
            lines.append(f"[warn] {w}")

    return "\n".join(lines)

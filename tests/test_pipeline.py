"""
Pipeline integration tests.

These tests cover the full stack:
  section_detector → NER (vocabulary fallback) → classifier → pipeline

The vocabulary-fallback NER is used here so the tests run without
any scispacy model download.
"""

import pytest
from src.section_detector import detect_sections, NoteSection
from src.ner import extract_entities
from src.pipeline import process_note, PipelineResult
from src.sentence_splitter import split_sentences, find_sentence_context

# ---------------------------------------------------------------------------
# Shared sample note
# ---------------------------------------------------------------------------

SAMPLE_NOTE = """\
Chief Complaint:
Shortness of breath and fatigue for 3 days.

History of Present Illness:
67-year-old female presenting with worsening dyspnea. She reports progressive
shortness of breath over the past 3 days. Denies chest pain or fever.

Past Medical History:
Hypertension, type 2 diabetes mellitus (diagnosed 5 years ago),
h/o pneumonia (resolved last year), atrial fibrillation controlled on medication.

Past Surgical History:
s/p appendectomy 10 years ago.

Assessment:
Possible heart failure exacerbation. Rule out pulmonary embolism.
Hypertension well-controlled. Diabetes stable.
"""


# ---------------------------------------------------------------------------
# Section detector tests
# ---------------------------------------------------------------------------

class TestSectionDetector:

    def test_detects_known_sections(self):
        sections = detect_sections(SAMPLE_NOTE)
        names = [s.name for s in sections]
        assert "chief_complaint" in names
        assert "history_of_present_illness" in names
        assert "past_medical_history" in names
        assert "assessment" in names

    def test_pmh_has_resolved_prior(self):
        sections = detect_sections(SAMPLE_NOTE)
        pmh = next(s for s in sections if s.name == "past_medical_history")
        assert pmh.status_prior == "resolved"

    def test_hpi_has_ongoing_prior(self):
        sections = detect_sections(SAMPLE_NOTE)
        hpi = next(s for s in sections if s.name == "history_of_present_illness")
        assert hpi.status_prior == "ongoing"

    def test_assessment_has_ongoing_prior(self):
        sections = detect_sections(SAMPLE_NOTE)
        assessment = next(s for s in sections if s.name == "assessment")
        assert assessment.status_prior == "ongoing"

    def test_no_headers_returns_unknown(self):
        sections = detect_sections("Patient has hypertension and chest pain.")
        assert len(sections) == 1
        assert sections[0].name == "unknown"

    def test_section_text_is_nonempty(self):
        sections = detect_sections(SAMPLE_NOTE)
        for s in sections:
            if s.name != "unknown":
                assert s.text.strip(), f"Section '{s.name}' has empty text"

    def test_family_history_gets_no_prior(self):
        note = "Family History:\nMother had hypertension and diabetes.\n"
        sections = detect_sections(note)
        fh = next((s for s in sections if s.name == "family_history"), None)
        assert fh is not None
        assert fh.status_prior is None

    def test_handles_abbreviated_headers(self):
        note = "PMH:\nDiabetes, hypertension.\nHPI:\nChest pain since morning.\n"
        sections = detect_sections(note)
        names = [s.name for s in sections]
        assert "past_medical_history" in names
        assert "history_of_present_illness" in names


# ---------------------------------------------------------------------------
# NER (vocabulary fallback) tests
# ---------------------------------------------------------------------------

class TestNER:

    def test_extracts_known_conditions(self):
        text = "Patient has hypertension and diabetes."
        entities = extract_entities(text)
        terms = [e.text.lower() for e in entities]
        assert any("hypertension" in t for t in terms)
        assert any("diabetes" in t for t in terms)

    def test_entity_offsets_are_correct(self):
        text = "Patient has hypertension."
        entities = extract_entities(text)
        hyp = next(e for e in entities if "hypertension" in e.text.lower())
        assert text[hyp.start:hyp.end].lower() == "hypertension"

    def test_no_entities_on_empty_text(self):
        assert extract_entities("") == []

    def test_handles_abbreviated_conditions(self):
        # After normalizer expands abbreviations, vocabulary should pick up HTN
        from src.normalizer import normalize
        text, _ = normalize("Patient has HTN and DM.")
        entities = extract_entities(text)
        terms = [e.text.lower() for e in entities]
        assert any("hypertension" in t for t in terms)
        assert any("diabetes" in t for t in terms)


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_returns_pipeline_result(self):
        result = process_note(SAMPLE_NOTE)
        assert isinstance(result, PipelineResult)

    def test_finds_conditions(self):
        result = process_note(SAMPLE_NOTE)
        assert len(result.conditions) > 0

    def test_condition_names_are_strings(self):
        result = process_note(SAMPLE_NOTE)
        for c in result.conditions:
            assert isinstance(c.condition, str)
            assert c.condition.strip()

    def test_all_statuses_are_valid(self):
        valid = {"ongoing", "resolved", "negated", "ambiguous"}
        result = process_note(SAMPLE_NOTE)
        for c in result.conditions:
            assert c.status in valid, f"Invalid status '{c.status}' for '{c.condition}'"

    def test_confidence_in_range(self):
        result = process_note(SAMPLE_NOTE)
        for c in result.conditions:
            assert 0.0 <= c.confidence <= 1.0

    def test_sections_detected(self):
        result = process_note(SAMPLE_NOTE)
        assert len(result.sections_found) > 0

    def test_empty_note_returns_warning(self):
        result = process_note("")
        assert len(result.conditions) == 0
        assert any("empty" in w.lower() for w in result.warnings)

    def test_family_history_skipped(self):
        note = (
            "Family History:\nMother had hypertension and diabetes.\n"
            "Assessment:\nPatient has hypertension.\n"
        )
        result = process_note(note)
        # Family history is skipped; conditions come only from Assessment
        for c in result.conditions:
            assert c.section != "family_history"

    def test_pmh_conditions_lean_resolved(self):
        note = "Past Medical History:\nDiabetes, hypertension, pneumonia.\n"
        result = process_note(note)
        # All conditions from PMH should be resolved (prior override kicks in
        # when classifier confidence is low on bare condition mentions)
        pmh_results = [c for c in result.conditions if c.section == "past_medical_history"]
        if pmh_results:
            resolved_count = sum(1 for c in pmh_results if c.status == "resolved")
            assert resolved_count > 0, "Expected at least one resolved condition from PMH"

    def test_no_duplicate_conditions(self):
        # Same condition in two sections — only first occurrence kept
        note = (
            "Past Medical History:\nHypertension.\n"
            "Assessment:\nHypertension well-controlled.\n"
        )
        result = process_note(note)
        conditions_lower = [c.condition.lower() for c in result.conditions]
        hypertension_count = sum(1 for c in conditions_lower if "hypertension" in c)
        assert hypertension_count == 1

    def test_ner_method_reported(self):
        result = process_note(SAMPLE_NOTE)
        assert result.ner_method in ("scispacy", "vocabulary")


# ---------------------------------------------------------------------------
# Sentence splitter tests
# ---------------------------------------------------------------------------

class TestSentenceSplitter:

    def test_splits_basic_sentences(self):
        text = "Patient has fever. Denies chest pain. No cough."
        sents = split_sentences(text)
        assert len(sents) == 3
        assert sents[0].text == "Patient has fever."
        assert sents[1].text == "Denies chest pain."
        assert sents[2].text == "No cough."

    def test_does_not_split_on_dr(self):
        text = "Dr. Smith examined the patient. He has hypertension."
        sents = split_sentences(text)
        assert len(sents) == 2
        assert "Dr. Smith" in sents[0].text

    def test_does_not_split_on_decimal_numbers(self):
        text = "Patient weighs 3.5 kg. She has diabetes."
        sents = split_sentences(text)
        assert len(sents) == 2
        assert "3.5 kg" in sents[0].text

    def test_does_not_split_on_mg_dose(self):
        text = "Patient takes metformin 500 mg. She has diabetes."
        sents = split_sentences(text)
        assert len(sents) == 2

    def test_character_offsets_are_correct(self):
        text = "No fever. Patient has diabetes."
        sents = split_sentences(text)
        for s in sents:
            assert text[s.start:s.end].strip() == s.text

    def test_empty_text_returns_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence_no_split(self):
        text = "Patient has hypertension."
        sents = split_sentences(text)
        assert len(sents) == 1
        assert sents[0].text == text

    # KEY TEST — the exact pollution bug this fix addresses
    def test_negation_does_not_bleed_across_sentences(self):
        """
        'No fever. Patient has diabetes.'
        With the old char window, classifying 'diabetes' could pick up 'No'
        from the previous sentence. With sentence-aware context it must not.
        """
        text = "No fever. Patient has diabetes."
        sents = split_sentences(text)

        # Find which sentence contains 'diabetes'
        diabetes_start = text.index("diabetes")
        diabetes_end = diabetes_start + len("diabetes")
        ctx = find_sentence_context(sents, diabetes_start, diabetes_end)

        assert "No fever" not in ctx, (
            "Negation from a previous sentence leaked into diabetes context"
        )
        assert "diabetes" in ctx

    def test_negation_stays_in_its_own_sentence(self):
        """The sentence with 'Denies' should be its own clean context."""
        text = "Patient has hypertension. Denies chest pain. No fever."
        sents = split_sentences(text)

        cp_start = text.index("chest pain")
        cp_end = cp_start + len("chest pain")
        ctx = find_sentence_context(sents, cp_start, cp_end)

        assert "Denies" in ctx
        assert "hypertension" not in ctx  # previous sentence must not bleed in
        assert "No fever" not in ctx      # following sentence must not bleed in

    def test_multiline_section_text(self):
        text = (
            "67-year-old female presenting with worsening dyspnea.\n"
            "She reports fatigue for 3 days.\n"
            "Denies chest pain or fever."
        )
        sents = split_sentences(text)
        assert len(sents) == 3

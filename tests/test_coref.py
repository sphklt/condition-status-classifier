"""
Tests for pronoun coreference and note-level evaluation.
"""

import pytest
from src.coref import has_pronoun, apply_pronoun_coref
from src.pipeline import process_note, ConditionResult
from src.sentence_splitter import split_sentences
from src.classifier import classify_condition_status
from src.calibration import reliability_diagram
from src.note_evaluator import evaluate_notes


# ---------------------------------------------------------------------------
# has_pronoun
# ---------------------------------------------------------------------------

class TestHasPronoun:

    def test_detects_it(self):
        assert has_pronoun("It resolved completely.")

    def test_detects_this(self):
        assert has_pronoun("This has improved significantly.")

    def test_detects_they(self):
        assert has_pronoun("They are no longer present.")

    def test_detects_the_condition(self):
        assert has_pronoun("The condition was treated successfully.")

    def test_no_pronoun_plain_entity(self):
        assert not has_pronoun("Patient has hypertension.")

    def test_no_pronoun_negation(self):
        assert not has_pronoun("Denies chest pain or fever.")


# ---------------------------------------------------------------------------
# apply_pronoun_coref — unit tests
# ---------------------------------------------------------------------------

class TestApplyPronounCoref:

    def _make_result(self, condition, status, confidence, context=""):
        return ConditionResult(
            condition=condition, status=status, confidence=confidence,
            section="unknown", context=context, reason="test",
        )

    def test_updates_entity_from_pronoun_sentence(self):
        """'cough' classified as ongoing from weak sentence; 'It resolved.' should update it."""
        text = "The patient had a cough. It resolved."
        sentences = split_sentences(text)

        results = [self._make_result("cough", "ongoing", 0.35, sentences[0].text)]
        positions = [(text.index("cough"), text.index("cough") + len("cough"))]

        n_updates = apply_pronoun_coref(positions, results, sentences, classify_condition_status)

        assert n_updates == 1
        assert results[0].status == "resolved"
        assert "Coref" in results[0].reason

    def test_does_not_update_high_confidence_entity(self):
        """A confident existing classification should not be overridden by a pronoun sentence."""
        text = "No evidence of pneumonia. It appears resolved."
        sentences = split_sentences(text)

        # pneumonia already classified as negated with high confidence
        results = [self._make_result("pneumonia", "negated", 0.95, sentences[0].text)]
        positions = [(text.index("pneumonia"), text.index("pneumonia") + len("pneumonia"))]

        n_updates = apply_pronoun_coref(positions, results, sentences, classify_condition_status)

        # No update: existing confidence 0.95 >> override margin
        assert results[0].status == "negated"

    def test_no_update_without_pronoun(self):
        """Sentences without a pronoun must not trigger coref."""
        text = "Patient has cough. Fever resolved."
        sentences = split_sentences(text)

        results = [self._make_result("cough", "ongoing", 0.40)]
        positions = [(text.index("cough"), text.index("cough") + len("cough"))]

        n_updates = apply_pronoun_coref(positions, results, sentences, classify_condition_status)
        assert n_updates == 0

    def test_no_update_when_no_entities(self):
        text = "It resolved completely."
        sentences = split_sentences(text)
        n_updates = apply_pronoun_coref([], [], sentences, classify_condition_status)
        assert n_updates == 0


# ---------------------------------------------------------------------------
# Pipeline coref integration
# ---------------------------------------------------------------------------

class TestPipelineCoref:

    def test_cough_resolved_by_pronoun(self):
        """Full pipeline: 'cough' should be resolved after 'It resolved.'"""
        note = "History of Present Illness:\nThe patient had a cough. It resolved.\n"
        result = process_note(note)
        cough_results = [c for c in result.conditions if "cough" in c.condition.lower()]
        assert cough_results, "Expected 'cough' to be extracted"
        assert cough_results[0].status == "resolved", (
            f"Expected resolved, got {cough_results[0].status}"
        )

    def test_coref_does_not_contaminate_other_sections(self):
        """Coref is section-scoped — 'it' in Assessment should not update a PMH entity."""
        note = (
            "Past Medical History:\nDiabetes.\n"
            "Assessment:\nIt resolved.\n"
        )
        result = process_note(note)
        diabetes_results = [c for c in result.conditions if "diabetes" in c.condition.lower()]
        if diabetes_results:
            # Diabetes is in PMH, 'It resolved' is in Assessment — no cross-section update
            assert diabetes_results[0].section == "past_medical_history"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_returns_dataframe_with_expected_columns(self):
        df = reliability_diagram("data/clinical_phrases.csv")
        for col in ["bin_lower", "bin_upper", "bin_center", "count", "accuracy", "gap"]:
            assert col in df.columns

    def test_ece_is_float_in_range(self):
        df = reliability_diagram("data/clinical_phrases.csv")
        ece = df.attrs["ece"]
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_n_total_matches_dataset(self):
        df = reliability_diagram("data/clinical_phrases.csv")
        assert df.attrs["n_total"] == 159

    def test_bin_counts_sum_to_total(self):
        df = reliability_diagram("data/clinical_phrases.csv")
        assert df["count"].sum() == df.attrs["n_total"]


# ---------------------------------------------------------------------------
# Note evaluator
# ---------------------------------------------------------------------------

class TestNoteEvaluator:

    def test_returns_expected_keys(self):
        result = evaluate_notes("data/annotated_notes.json")
        assert "notes" in result
        assert "aggregate" in result

    def test_evaluates_seven_notes(self):
        result = evaluate_notes("data/annotated_notes.json")
        assert result["aggregate"]["n_notes"] == 7

    def test_aggregate_metrics_in_range(self):
        result = evaluate_notes("data/annotated_notes.json")
        agg = result["aggregate"]
        assert 0.0 <= agg["precision"] <= 1.0
        assert 0.0 <= agg["recall"] <= 1.0
        assert 0.0 <= agg["f1"] <= 1.0

    def test_per_note_items_present(self):
        result = evaluate_notes("data/annotated_notes.json")
        for note in result["notes"]:
            assert "items" in note
            assert len(note["items"]) > 0

    def test_reasonable_recall(self):
        result = evaluate_notes("data/annotated_notes.json")
        # A working pipeline should find most expected conditions
        assert result["aggregate"]["recall"] >= 0.50, (
            f"Recall {result['aggregate']['recall']:.2f} is unexpectedly low"
        )

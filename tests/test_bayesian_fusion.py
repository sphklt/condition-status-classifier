"""
Tests for Bayesian evidence fusion (src/bayesian_fusion.py).
"""

import pytest
from src.bayesian_fusion import fuse, evaluate_fusion, LABELS


# ---------------------------------------------------------------------------
# Return schema
# ---------------------------------------------------------------------------

class TestFuseSchema:

    def test_returns_required_keys(self):
        result = fuse("Patient has diabetes.")
        for key in ("status", "confidence", "posterior", "entropy", "cue",
                    "log_scores", "calibrated_confidence", "signals"):
            assert key in result, f"Missing key: {key}"

    def test_status_is_valid_label(self):
        result = fuse("Patient has diabetes.")
        assert result["status"] in LABELS

    def test_posterior_sums_to_one(self):
        result = fuse("No evidence of fever.")
        total = sum(result["posterior"].values())
        assert abs(total - 1.0) < 0.01, f"Posterior sums to {total}"

    def test_posterior_has_all_labels(self):
        result = fuse("History of hypertension.")
        assert set(result["posterior"].keys()) == set(LABELS)

    def test_confidence_equals_map_posterior(self):
        result = fuse("Patient presents with chest pain.")
        map_prob = result["posterior"][result["status"]]
        assert abs(result["confidence"] - map_prob) < 0.001

    def test_confidence_in_range(self):
        result = fuse("Possible pneumonia.")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_entropy_in_range(self):
        result = fuse("Patient has no fever.")
        assert 0.0 <= result["entropy"] <= 2.0

    def test_calibrated_confidence_in_range(self):
        result = fuse("Resolved after treatment.")
        assert 0.0 <= result["calibrated_confidence"] <= 1.0

    def test_empty_text_returns_ambiguous(self):
        result = fuse("")
        assert result["status"] == "ambiguous"
        assert result["entropy"] == 2.0


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

class TestFuseClassification:

    def test_negation_strong_cue(self):
        result = fuse("Patient has no fever.")
        assert result["status"] == "negated"

    def test_negation_multi_word(self):
        result = fuse("No evidence of chest pain.")
        assert result["status"] == "negated"

    def test_resolved_history_of(self):
        result = fuse("History of hypertension.")
        assert result["status"] == "resolved"

    def test_resolved_no_longer(self):
        result = fuse("No longer has headache.")
        assert result["status"] == "resolved"

    def test_ongoing_presents_with(self):
        result = fuse("Patient presents with chest pain.")
        assert result["status"] == "ongoing"

    def test_ongoing_currently(self):
        result = fuse("Currently experiencing shortness of breath.")
        assert result["status"] == "ongoing"

    def test_ambiguous_possible(self):
        result = fuse("Possible pneumonia.")
        assert result["status"] == "ambiguous"

    def test_ambiguous_cannot_rule_out(self):
        result = fuse("Cannot rule out pulmonary embolism.")
        assert result["status"] == "ambiguous"

    def test_status_post(self):
        result = fuse("s/p appendectomy.")
        assert result["status"] == "resolved"


# ---------------------------------------------------------------------------
# Posterior distribution properties
# ---------------------------------------------------------------------------

class TestPosteriorProperties:

    def test_strong_negation_peaks_negated(self):
        result = fuse("No evidence of chest pain.")
        assert result["posterior"]["negated"] > 0.80

    def test_strong_resolved_peaks_resolved(self):
        result = fuse("No longer has headache.")
        assert result["posterior"]["resolved"] > 0.90

    def test_strong_ongoing_peaks_ongoing(self):
        result = fuse("Currently experiencing worsening chest pain.")
        assert result["posterior"]["ongoing"] > 0.90

    def test_strong_ambiguous_peaks_ambiguous(self):
        result = fuse("Cannot rule out pulmonary embolism.")
        assert result["posterior"]["ambiguous"] > 0.80

    def test_competing_signals_spread_posterior(self):
        # "History of" (resolved) + "currently" (ongoing) — conflict
        result = fuse("History of hypertension, currently on medication.")
        # Neither label should dominate (posterior spread > 0 for both)
        assert result["posterior"]["resolved"] > 0.05
        assert result["posterior"]["ongoing"] > 0.05


# ---------------------------------------------------------------------------
# Entropy as uncertainty signal
# ---------------------------------------------------------------------------

class TestEntropy:

    def test_high_confidence_low_entropy(self):
        result = fuse("No evidence of fever.")
        assert result["entropy"] < 1.0, (
            f"Expected low entropy, got {result['entropy']}"
        )

    def test_conflicting_signals_raise_entropy(self):
        # Equal-weight conflict: "history of" (resolved, 0.95) vs
        # "poorly controlled" (ongoing, 0.95) → posterior spreads across both
        r_clean    = fuse("History of hypertension.")
        r_conflict = fuse("History of poorly controlled hypertension.")
        assert r_conflict["entropy"] > r_clean["entropy"]

    def test_maximum_entropy_is_two_bits(self):
        # Empty text → uniform posterior → max entropy
        result = fuse("")
        assert abs(result["entropy"] - 2.0) < 0.001


# ---------------------------------------------------------------------------
# Section prior effect
# ---------------------------------------------------------------------------

class TestSectionPrior:

    def test_pmh_section_boosts_resolved(self):
        phrase = "Hypertension."
        r_pmh     = fuse(phrase, section="past_medical_history")
        r_unknown = fuse(phrase, section="unknown")
        # PMH prior strongly favours resolved → its posterior probability higher
        assert r_pmh["posterior"]["resolved"] > r_unknown["posterior"]["resolved"]

    def test_hpi_section_boosts_ongoing(self):
        phrase = "Hypertension."
        r_hpi     = fuse(phrase, section="hpi")
        r_unknown = fuse(phrase, section="unknown")
        assert r_hpi["posterior"]["ongoing"] > r_unknown["posterior"]["ongoing"]

    def test_pmh_section_classifies_ambiguous_entity_as_resolved(self):
        result = fuse("Diabetes.", section="past_medical_history")
        assert result["status"] == "resolved"


# ---------------------------------------------------------------------------
# evaluate_fusion
# ---------------------------------------------------------------------------

class TestEvaluateFusion:

    def test_returns_required_keys(self):
        result = evaluate_fusion("data/clinical_phrases.csv")
        for key in ("accuracy", "correct", "n", "ece", "per_label"):
            assert key in result, f"Missing key: {key}"

    def test_n_matches_dataset(self):
        result = evaluate_fusion("data/clinical_phrases.csv")
        assert result["n"] == 127

    def test_accuracy_in_range(self):
        result = evaluate_fusion("data/clinical_phrases.csv")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_reasonable_accuracy(self):
        result = evaluate_fusion("data/clinical_phrases.csv")
        assert result["accuracy"] >= 0.80, (
            f"Accuracy {result['accuracy']:.2%} is unexpectedly low"
        )

    def test_ece_in_range(self):
        result = evaluate_fusion("data/clinical_phrases.csv")
        assert 0.0 <= result["ece"] <= 1.0

    def test_per_label_has_all_labels(self):
        result = evaluate_fusion("data/clinical_phrases.csv")
        for label in LABELS:
            assert label in result["per_label"], f"Missing label: {label}"

    def test_per_label_n_sums_to_total(self):
        result = evaluate_fusion("data/clinical_phrases.csv")
        total = sum(v["n"] for v in result["per_label"].values())
        assert total == result["n"]

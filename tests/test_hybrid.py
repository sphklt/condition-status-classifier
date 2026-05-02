"""
Tests for the hybrid classifier (src/hybrid.py).
"""

from src.hybrid import classify, evaluate_triage, TRIAGE_THRESHOLD
from src.bayesian_fusion import LABELS


# ---------------------------------------------------------------------------
# Return schema
# ---------------------------------------------------------------------------

class TestClassifySchema:

    def test_returns_required_keys(self):
        result = classify("Patient has no fever.")
        for key in ("status", "confidence", "calibrated_confidence",
                    "posterior", "entropy", "runner_up",
                    "agreement", "triage_flag", "triage_reason",
                    "rule_reason", "rule_cue", "bayes_status", "signals"):
            assert key in result, f"Missing key: {key}"

    def test_status_is_valid_label(self):
        result = classify("History of hypertension.")
        assert result["status"] in LABELS

    def test_posterior_sums_to_one(self):
        result = classify("Possible pneumonia.")
        assert abs(sum(result["posterior"].values()) - 1.0) < 0.01

    def test_runner_up_is_tuple(self):
        result = classify("Patient has no fever.")
        label, prob = result["runner_up"]
        assert label in LABELS
        assert 0.0 <= prob <= 1.0

    def test_runner_up_is_not_map_label(self):
        result = classify("No evidence of chest pain.")
        map_label = result["status"]
        runner_label = result["runner_up"][0]
        assert runner_label != map_label

    def test_entropy_in_range(self):
        result = classify("Resolved after treatment.")
        assert 0.0 <= result["entropy"] <= 2.0

    def test_entropy_not_negative(self):
        # High-confidence predictions should show 0.0, not −0.0
        result = classify("No evidence of pneumonia.")
        assert result["entropy"] >= 0.0

    def test_empty_text_is_flagged(self):
        result = classify("")
        assert result["triage_flag"] is True

    def test_agreement_is_bool(self):
        result = classify("History of hypertension.")
        assert isinstance(result["agreement"], bool)

    def test_triage_flag_is_bool(self):
        result = classify("History of hypertension.")
        assert isinstance(result["triage_flag"], bool)


# ---------------------------------------------------------------------------
# Status (rule-based prediction)
# ---------------------------------------------------------------------------

class TestClassifyStatus:

    def test_negation(self):
        assert classify("Patient has no fever.")["status"] == "negated"

    def test_resolved(self):
        assert classify("History of hypertension.")["status"] == "resolved"

    def test_ongoing(self):
        assert classify("Patient presents with chest pain.")["status"] == "ongoing"

    def test_ambiguous(self):
        assert classify("Possible pneumonia.")["status"] == "ambiguous"


# ---------------------------------------------------------------------------
# Triage flag logic
# ---------------------------------------------------------------------------

class TestTriageFlag:

    def test_clear_negation_not_flagged(self):
        # Strong, unambiguous negation — should be auto-approved
        result = classify("No evidence of chest pain.")
        assert result["triage_flag"] is False

    def test_clear_resolved_not_flagged(self):
        result = classify("No longer has headache.")
        assert result["triage_flag"] is False

    def test_flagged_when_entropy_above_threshold(self):
        # Entropy threshold is TRIAGE_THRESHOLD — phrases with no strong cues
        # fall back to near-uniform posterior → flagged
        result = classify("The patient mentioned something.")
        if result["entropy"] > TRIAGE_THRESHOLD:
            assert result["triage_flag"] is True

    def test_flagged_when_systems_disagree(self):
        # When agreement=False the flag must be True regardless of entropy
        result = classify("My back pain is gone.")
        if not result["agreement"]:
            assert result["triage_flag"] is True

    def test_triage_reason_empty_when_not_flagged(self):
        result = classify("No evidence of chest pain.")
        if not result["triage_flag"]:
            assert result["triage_reason"] == ""

    def test_triage_reason_nonempty_when_flagged(self):
        result = classify("")
        assert result["triage_flag"] is True
        assert result["triage_reason"] != ""


# ---------------------------------------------------------------------------
# Agreement between rule-based and Bayesian systems
# ---------------------------------------------------------------------------

class TestAgreement:

    def test_agreement_true_for_strong_cues(self):
        # Both systems should agree on "No evidence of" (weight=1.0)
        result = classify("No evidence of pneumonia.")
        assert result["agreement"] is True

    def test_agreement_true_for_strong_resolved(self):
        result = classify("Asthma resolved after treatment.")
        assert result["agreement"] is True

    def test_agreement_flag_implies_triage(self):
        # Whenever agreement=False, triage_flag must be True
        for phrase in [
            "My back pain is gone.",
            "The fever broke this morning.",
            "Troponin negative.",
        ]:
            r = classify(phrase)
            if not r["agreement"]:
                assert r["triage_flag"] is True, (
                    f"agreement=False but triage_flag=False for: {phrase}"
                )


# ---------------------------------------------------------------------------
# evaluate_triage
# ---------------------------------------------------------------------------

class TestEvaluateTriage:

    def test_returns_required_keys(self):
        result = evaluate_triage("data/clinical_phrases.csv")
        assert "results" in result
        assert "per_phrase" in result
        assert "summary" in result

    def test_results_has_threshold_rows(self):
        result = evaluate_triage("data/clinical_phrases.csv",
                                 thresholds=[0.5, 1.0, 1.5, 2.0])
        assert len(result["results"]) == 4

    def test_summary_n_matches_dataset(self):
        result = evaluate_triage("data/clinical_phrases.csv")
        assert result["summary"]["n"] == 159

    def test_metrics_in_range(self):
        result = evaluate_triage("data/clinical_phrases.csv",
                                 thresholds=[TRIAGE_THRESHOLD])
        row = result["results"][0]
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"]    <= 1.0
        assert 0.0 <= row["f1"]        <= 1.0

    def test_low_threshold_catches_all_errors(self):
        # At threshold 0.5 (very low) — most wrong predictions are flagged
        result = evaluate_triage("data/clinical_phrases.csv",
                                 thresholds=[0.5])
        row = result["results"][0]
        # Recall should be high (close to 1.0)
        assert row["recall"] >= 0.80

    def test_per_phrase_length_matches_n(self):
        result = evaluate_triage("data/clinical_phrases.csv")
        assert len(result["per_phrase"]) == result["summary"]["n"]

    def test_per_phrase_has_required_keys(self):
        result = evaluate_triage("data/clinical_phrases.csv")
        for phrase in result["per_phrase"][:3]:
            for key in ("text", "gold", "pred", "correct", "entropy",
                        "agreement", "flagged"):
                assert key in phrase, f"Missing key '{key}' in per_phrase entry"

    def test_default_threshold_recall_is_perfect(self):
        # At default threshold (1.2), every wrong prediction should be flagged
        result = evaluate_triage("data/clinical_phrases.csv",
                                 thresholds=[TRIAGE_THRESHOLD])
        row = result["results"][0]
        assert row["recall"] == 1.0, (
            f"Expected 100% recall at default threshold, got {row['recall']:.0%}"
        )

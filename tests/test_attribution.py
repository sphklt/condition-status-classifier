"""
Tests for attribution-aware confidence (src/attribution.py).
"""

from src.attribution import (
    AttributionSignature,
    extract_attribution,
    attribution_to_llr,
    SOURCES,
)
from src.bayesian_fusion import fuse, LABELS


# ---------------------------------------------------------------------------
# extract_attribution — source detection
# ---------------------------------------------------------------------------

class TestAttributionExtraction:

    def test_no_attribution_plain_clinician(self):
        assert extract_attribution("Patient has hypertension.").source == "none"

    def test_no_attribution_direct_statement(self):
        assert extract_attribution("Chest pain is ongoing.").source == "none"

    # Patient report

    def test_patient_reports(self):
        assert extract_attribution("Patient reports chest pain.").source == "patient_report"

    def test_patient_states(self):
        assert extract_attribution("Patient states she has headache.").source == "patient_report"

    def test_patient_endorses(self):
        assert extract_attribution("Patient endorses fatigue.").source == "patient_report"

    def test_patient_complains_of(self):
        assert extract_attribution("Patient complains of nausea.").source == "patient_report"

    def test_per_patient(self):
        assert extract_attribution("Per patient, he has diabetes.").source == "patient_report"

    # Patient hedge

    def test_patient_thinks(self):
        assert extract_attribution("Patient thinks she might have diabetes.").source == "patient_hedge"

    def test_patient_believes(self):
        assert extract_attribution("Patient believes he has hypertension.").source == "patient_hedge"

    def test_patient_suspects(self):
        assert extract_attribution("Patient suspects she could have depression.").source == "patient_hedge"

    def test_patient_hedge_priority_over_report(self):
        # "Patient thinks she reports" — hedge takes priority
        assert extract_attribution("Patient thinks she reports having anxiety.").source == "patient_hedge"

    # Family report

    def test_family_reports(self):
        assert extract_attribution("Family reports patient had seizures.").source == "family_report"

    def test_wife_states(self):
        assert extract_attribution("Wife states husband had chest pain.").source == "family_report"

    def test_family_history_no_verb_no_fire(self):
        # "family history" with no report verb must NOT fire
        assert extract_attribution("Family history of diabetes.").source == "none"

    def test_family_history_of_no_fire(self):
        assert extract_attribution("Strong family history of heart disease.").source == "none"

    # Record

    def test_per_records(self):
        assert extract_attribution("Per records, hypertension noted.").source == "record"

    def test_per_chart(self):
        assert extract_attribution("Per chart, asthma.").source == "record"

    def test_medical_records_show(self):
        assert extract_attribution("Medical records show diabetes mellitus.").source == "record"

    def test_records_document(self):
        assert extract_attribution("Records document prior depression.").source == "record"

    def test_per_ehr(self):
        assert extract_attribution("Per EHR, atrial fibrillation.").source == "record"

    def test_records_indicate(self):
        assert extract_attribution("Records indicate migraine.").source == "record"

    # Clinician hedge

    def test_we_think(self):
        assert extract_attribution("We think this represents heart failure.").source == "clinician_hedge"

    def test_we_believe(self):
        assert extract_attribution("We believe this is pneumonia.").source == "clinician_hedge"

    def test_it_is_thought(self):
        assert extract_attribution("It is thought to be atrial fibrillation.").source == "clinician_hedge"

    def test_appears_consistent_with(self):
        assert extract_attribution("Appears consistent with COPD.").source == "clinician_hedge"


# ---------------------------------------------------------------------------
# attribution_to_llr — LLR direction
# ---------------------------------------------------------------------------

class TestAttributionLLR:

    def test_none_source_all_zeros(self):
        sig = AttributionSignature(source="none")
        llrs = attribution_to_llr(sig, LABELS)
        assert all(v == 0.0 for v in llrs.values())

    def test_record_boosts_resolved(self):
        sig = AttributionSignature(source="record")
        llrs = attribution_to_llr(sig, LABELS)
        assert llrs["resolved"] > 0.0
        assert llrs["resolved"] > llrs["ongoing"]

    def test_patient_hedge_boosts_ambiguous(self):
        sig = AttributionSignature(source="patient_hedge")
        llrs = attribution_to_llr(sig, LABELS)
        assert llrs["ambiguous"] > 0.0
        assert llrs["ambiguous"] > llrs["ongoing"]

    def test_family_report_boosts_ambiguous(self):
        sig = AttributionSignature(source="family_report")
        llrs = attribution_to_llr(sig, LABELS)
        assert llrs["ambiguous"] > 0.0

    def test_clinician_hedge_boosts_ambiguous(self):
        sig = AttributionSignature(source="clinician_hedge")
        llrs = attribution_to_llr(sig, LABELS)
        assert llrs["ambiguous"] > 0.0

    def test_patient_report_mild_ambiguous_shift(self):
        sig = AttributionSignature(source="patient_report")
        llrs = attribution_to_llr(sig, LABELS)
        assert llrs["ambiguous"] > 0.0
        # patient_report is weaker than patient_hedge
        sig_h = AttributionSignature(source="patient_hedge")
        llrs_h = attribution_to_llr(sig_h, LABELS)
        assert llrs["ambiguous"] < llrs_h["ambiguous"]

    def test_record_reduces_ongoing(self):
        sig = AttributionSignature(source="record")
        llrs = attribution_to_llr(sig, LABELS)
        assert llrs["ongoing"] < 0.0

    def test_all_sources_covered(self):
        for source in SOURCES:
            if source == "none":
                continue
            sig = AttributionSignature(source=source)
            llrs = attribution_to_llr(sig, LABELS)
            assert set(llrs.keys()) == set(LABELS)


# ---------------------------------------------------------------------------
# Fusion integration — attribution changes fuse() output
# ---------------------------------------------------------------------------

class TestAttributionFusionIntegration:

    def test_record_phrase_resolves(self):
        # No other resolved cue — attribution (record) provides the resolved signal
        result = fuse("Records show hypertension.")
        assert result["status"] == "resolved", (
            f"Expected resolved for record-attributed phrase, got {result['status']}"
        )

    def test_per_records_resolves(self):
        result = fuse("Per records, hypertension.")
        assert result["status"] == "resolved"

    def test_per_chart_resolves(self):
        result = fuse("Per chart, asthma.")
        assert result["status"] == "resolved"

    def test_medical_records_document_resolves(self):
        result = fuse("Medical records document diabetes mellitus.")
        assert result["status"] == "resolved"

    def test_record_attribution_in_signals(self):
        result = fuse("Records show hypertension.")
        assert result["signals"]["attribution"] == "record"

    def test_patient_hedge_shifts_toward_ambiguous(self):
        # Compare same condition with and without patient hedge
        r_direct = fuse("Patient has hypertension.")
        r_hedged = fuse("Patient thinks she has hypertension.")
        assert r_hedged["posterior"]["ambiguous"] > r_direct["posterior"]["ambiguous"]

    def test_clinician_hedge_shifts_toward_ambiguous(self):
        r_direct = fuse("This is pneumonia.")
        r_hedged  = fuse("We think this is pneumonia.")
        assert r_hedged["posterior"]["ambiguous"] > r_direct["posterior"]["ambiguous"]

    def test_family_report_increases_ambiguous(self):
        r_direct = fuse("Patient has chest pain.")
        r_family = fuse("Family reports patient has chest pain.")
        assert r_family["posterior"]["ambiguous"] > r_direct["posterior"]["ambiguous"]

    def test_no_attribution_signal_is_none(self):
        result = fuse("Patient has diabetes.")
        assert result["signals"]["attribution"] is None

    def test_patient_report_attribution_in_signals(self):
        result = fuse("Patient reports chest pain.")
        assert result["signals"]["attribution"] == "patient_report"


# ---------------------------------------------------------------------------
# Safety — attribution does not corrupt strong cue predictions
# ---------------------------------------------------------------------------

class TestAttributionSafety:

    def test_patient_denies_stays_negated(self):
        # "Patient denies" fires patient_report attribution, but the
        # negation cue ("denies") is far stronger — must stay negated
        result = fuse("Patient denies fever.")
        assert result["status"] == "negated", (
            f"Patient denies should be negated, got {result['status']}"
        )

    def test_patient_reports_no_stays_negated(self):
        result = fuse("Patient reports no chest pain.")
        assert result["status"] == "negated"

    def test_strong_ongoing_cue_not_overridden_by_patient_report(self):
        # "currently" = weight 0.90, LLR >> patient_report LLR
        result = fuse("Patient reports currently worsening chest pain.")
        assert result["status"] == "ongoing"

    def test_strong_negation_not_overridden_by_record(self):
        # record attribution boosts resolved by +1.0 LLR
        # but "no evidence of" fires negation at weight ~1.0 (LLR ≈ 6.9)
        result = fuse("Records show no evidence of hypertension.")
        assert result["status"] == "negated"

    def test_family_history_no_verb_safe(self):
        # "family history of" must NOT trigger family_report attribution
        result1 = fuse("Family history of diabetes.")
        result2 = fuse("Diabetes.")
        # With no verb, family attribution should not fire
        assert result1["signals"]["attribution"] is None

    def test_history_of_resolved_not_overridden(self):
        # "history of" is a strong resolved cue; record attribution is additive
        result = fuse("Per records, history of hypertension.")
        assert result["status"] == "resolved"

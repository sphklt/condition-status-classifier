"""
Tests for TAM (Tense-Aspect-Modality) extraction and fusion integration.

Coverage
--------
TestTAMExtraction     — extract_tam() identifies the correct TAM component
TestTAMLLR            — tam_to_llr() contributes LLRs in the correct direction
TestTAMCompositionality — combined TAM components add independently
TestTAMFusionIntegration — TAM signals shift the Bayesian posterior correctly
TestTAMNegationSafety — TAM does not corrupt negation predictions
"""

from src.tam import TAMSignature, extract_tam, tam_to_llr
from src.bayesian_fusion import LABELS, fuse


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestTAMExtraction:

    # Aspect — most syntactically unambiguous

    def test_present_progressive(self):
        sig = extract_tam("Diabetes is worsening")
        assert sig.aspect == "progressive"
        assert sig.tense  == "present"

    def test_past_progressive(self):
        sig = extract_tam("Fever was increasing")
        assert sig.aspect == "progressive"
        assert sig.tense  == "past"

    def test_present_perfect(self):
        sig = extract_tam("Asthma has resolved")
        assert sig.aspect == "perfect"
        assert sig.tense  == "present"

    def test_past_perfect(self):
        sig = extract_tam("Hypertension had been well-controlled")
        assert sig.aspect == "past_perfect"
        assert sig.tense  == "past"

    def test_past_perfect_simple(self):
        sig = extract_tam("The infection had resolved before admission")
        assert sig.aspect == "past_perfect"

    # Modality

    def test_epistemic_weak_may(self):
        sig = extract_tam("Chest pain may indicate a cardiac event")
        assert sig.modal == "epistemic_weak"

    def test_epistemic_weak_might_is_weaker(self):
        sig_may   = extract_tam("Findings may suggest pneumonia")
        sig_might = extract_tam("Findings might suggest pneumonia")
        assert sig_may.modal   == "epistemic_weak"
        assert sig_might.modal == "epistemic_weak"
        assert sig_might.modal_strength < sig_may.modal_strength

    def test_epistemic_strong_probably(self):
        sig = extract_tam("Presentation is probably viral")
        assert sig.modal == "epistemic_strong"

    def test_epistemic_strong_consistent_with(self):
        sig = extract_tam("Findings consistent with heart failure")
        assert sig.modal == "epistemic_strong"

    def test_deontic_should(self):
        sig = extract_tam("Hypertension should be monitored closely")
        assert sig.modal == "deontic"

    def test_deontic_requires(self):
        sig = extract_tam("Diabetes requires ongoing management")
        assert sig.modal == "deontic"

    def test_negated_deontic_should_not(self):
        sig = extract_tam("Active infection should not be present")
        assert sig.modal == "negated_deontic"

    def test_negated_deontic_cannot(self):
        sig = extract_tam("Abscess cannot be seen on imaging")
        assert sig.modal == "negated_deontic"

    def test_cannot_exclude_is_epistemic_not_deontic(self):
        # "cannot exclude" = "we can't rule it out" → epistemic, not deontic
        sig = extract_tam("Cannot exclude pneumonia")
        assert sig.modal == "epistemic_weak"

    def test_conditional_would(self):
        sig = extract_tam("Symptoms would recur without treatment")
        assert sig.modal == "conditional"

    # Tense (conservative patterns only)

    def test_past_tense_resolved_verb(self):
        sig = extract_tam("Fever resolved after antibiotics")
        assert sig.tense == "past"

    def test_past_tense_was_stable(self):
        sig = extract_tam("Blood pressure was stable on admission")
        assert sig.tense == "past"

    def test_present_tense_remains(self):
        sig = extract_tam("Hypertension remains uncontrolled")
        assert sig.tense == "present"

    def test_present_tense_persists(self):
        sig = extract_tam("Cough persists despite treatment")
        assert sig.tense == "present"

    def test_future_will(self):
        sig = extract_tam("Condition will be reassessed next visit")
        assert sig.tense == "future"

    # No-signal cases

    def test_bare_noun_no_signal(self):
        assert extract_tam("Hypertension").has_signal() is False

    def test_empty_no_signal(self):
        assert extract_tam("").has_signal() is False

    def test_whitespace_no_signal(self):
        assert extract_tam("   ").has_signal() is False

    def test_simple_negation_phrase_no_signal(self):
        # "had no fever" must NOT trigger past tense (negation construction)
        sig = extract_tam("Patient had no fever")
        assert sig.tense == "unknown"

    def test_has_no_does_not_trigger_present_tense(self):
        # "has no" is a negation cue, not a present tense marker
        sig = extract_tam("Patient has no chest pain")
        assert sig.tense == "unknown"


# ---------------------------------------------------------------------------
# LLR direction and scaling
# ---------------------------------------------------------------------------

class TestTAMLLR:

    def test_progressive_boosts_ongoing(self):
        sig = TAMSignature(aspect="progressive")
        llr = tam_to_llr(sig, LABELS)
        assert llr["ongoing"] > 0
        assert llr["resolved"] < 0

    def test_past_perfect_boosts_resolved(self):
        sig = TAMSignature(aspect="past_perfect")
        llr = tam_to_llr(sig, LABELS)
        assert llr["resolved"] > 0
        assert llr["ongoing"] < 0

    def test_perfect_boosts_resolved(self):
        sig = TAMSignature(aspect="perfect")
        llr = tam_to_llr(sig, LABELS)
        assert llr["resolved"] > 0

    def test_past_tense_boosts_resolved(self):
        sig = TAMSignature(tense="past")
        llr = tam_to_llr(sig, LABELS)
        assert llr["resolved"] > 0
        assert llr["ongoing"] < 0

    def test_present_tense_boosts_ongoing(self):
        sig = TAMSignature(tense="present")
        llr = tam_to_llr(sig, LABELS)
        assert llr["ongoing"] > 0
        assert llr["resolved"] < 0

    def test_epistemic_weak_boosts_ambiguous(self):
        sig = TAMSignature(modal="epistemic_weak")
        llr = tam_to_llr(sig, LABELS)
        assert llr["ambiguous"] > llr["ongoing"]
        assert llr["ambiguous"] > llr["resolved"]
        assert llr["negated"] < 0

    def test_deontic_boosts_ongoing(self):
        sig = TAMSignature(modal="deontic")
        llr = tam_to_llr(sig, LABELS)
        assert llr["ongoing"] > 0
        assert llr["resolved"] < 0

    def test_negated_deontic_boosts_negated(self):
        sig = TAMSignature(modal="negated_deontic")
        llr = tam_to_llr(sig, LABELS)
        assert llr["negated"] > 0
        assert llr["ongoing"] < 0

    def test_no_signal_zero_llr(self):
        llr = tam_to_llr(TAMSignature(), LABELS)
        assert all(v == 0.0 for v in llr.values())

    def test_modal_strength_scales_llr(self):
        sig_strong = TAMSignature(modal="epistemic_weak", modal_strength=1.0)
        sig_weak   = TAMSignature(modal="epistemic_weak", modal_strength=0.5)
        llr_s = tam_to_llr(sig_strong, LABELS)
        llr_w = tam_to_llr(sig_weak,   LABELS)
        assert llr_s["ambiguous"] > llr_w["ambiguous"]


# ---------------------------------------------------------------------------
# Compositionality — combined components behave as expected
# ---------------------------------------------------------------------------

class TestTAMCompositionality:

    def test_epistemic_plus_progressive_both_contribute(self):
        # "might have been resolving"
        sig = TAMSignature(modal="epistemic_weak", aspect="progressive")
        llr = tam_to_llr(sig, LABELS)
        # Progressive pulls toward ongoing; epistemic pulls toward ambiguous.
        # Net: ongoing still positive (progressive dominates), ambiguous also positive.
        assert llr["ongoing"]   > 0
        assert llr["ambiguous"] > 0

    def test_past_perfect_plus_epistemic_high_entropy(self):
        # "might have resolved" — resolved cue + uncertainty
        sig = TAMSignature(modal="epistemic_weak", aspect="perfect")
        llr = tam_to_llr(sig, LABELS)
        # Perfect pulls toward resolved; epistemic pulls toward ambiguous.
        # Both resolved and ambiguous should be positive.
        assert llr["resolved"]  > 0
        assert llr["ambiguous"] > 0

    def test_past_tense_plus_progressive_partially_cancel(self):
        # "was worsening" — past + progressive (condition was active but may now be resolved)
        sig = TAMSignature(tense="past", aspect="progressive")
        llr = tam_to_llr(sig, LABELS)
        # Past pulls toward resolved; progressive pulls toward ongoing.
        # Net ongoing should be less than pure progressive, more than zero.
        pure_progressive = tam_to_llr(TAMSignature(aspect="progressive"), LABELS)
        assert 0 < llr["ongoing"] < pure_progressive["ongoing"]

    def test_strong_qualifier_increases_modal_strength(self):
        sig_plain  = extract_tam("May indicate pneumonia")
        sig_strong = extract_tam("Definitely may indicate pneumonia")
        assert sig_strong.modal_strength >= sig_plain.modal_strength

    def test_weak_qualifier_introduces_soft_epistemic(self):
        # "Mildly elevated" — "mildly" alone → soft epistemic (no explicit modal verb)
        sig = extract_tam("Mildly elevated white count")
        # Should introduce a soft epistemic signal (modal != "none")
        assert sig.modal == "epistemic_weak"
        assert sig.modal_strength <= 0.60


# ---------------------------------------------------------------------------
# Fusion integration — TAM shifts the Bayesian posterior correctly
# ---------------------------------------------------------------------------

class TestTAMFusionIntegration:

    def test_epistemic_raises_entropy_vs_certain(self):
        # "has been stable" (certain) vs "might have been stable" (epistemic)
        # Avoids weight=1.0 cues ("resolved") that would overwhelm the modal signal.
        certain   = fuse("Hypertension is stable")
        uncertain = fuse("Hypertension might be stable")
        assert uncertain["posterior"]["ambiguous"] > certain["posterior"]["ambiguous"]

    def test_present_progressive_classifies_ongoing(self):
        result = fuse("Symptoms are worsening rapidly")
        assert result["status"] == "ongoing"

    def test_past_perfect_classifies_resolved(self):
        result = fuse("The infection had resolved before admission")
        assert result["status"] == "resolved"

    def test_deontic_classifies_ongoing(self):
        # "should be monitored" — no explicit ongoing cue, TAM deontic → ongoing
        result = fuse("Blood pressure should be monitored")
        assert result["status"] == "ongoing"

    def test_epistemic_shifts_posterior_toward_ambiguous(self):
        # Adding "might" should increase the ambiguous mass in the posterior
        base      = fuse("Condition will worsen")
        epistemic = fuse("Condition might worsen")
        assert epistemic["posterior"]["ambiguous"] > base["posterior"]["ambiguous"]

    def test_tam_signal_recorded_in_output(self):
        result = fuse("Diabetes is progressing")
        assert result["signals"]["tam"] is not None
        assert result["signals"]["tam"]["aspect"] == "progressive"

    def test_no_tam_on_bare_condition(self):
        result = fuse("Hypertension")
        assert result["signals"]["tam"] is None

    def test_epistemic_raises_entropy_on_strong_cue(self):
        # "worsening" (weight=0.95) makes "are worsening" near-certain.
        # Adding "might" (epistemic_weak) should meaningfully increase entropy
        # by spreading posterior mass toward ambiguous.
        certain   = fuse("Symptoms are worsening")
        uncertain = fuse("Symptoms might be worsening")
        assert uncertain["entropy"] > certain["entropy"]


# ---------------------------------------------------------------------------
# Safety — TAM must not corrupt predictions that rely on strong cues
# ---------------------------------------------------------------------------

class TestTAMNegationSafety:

    def test_strong_negation_cue_not_flipped_by_present_tense(self):
        # "No evidence of pneumonia" — strong negation cue must win
        result = fuse("No evidence of pneumonia")
        assert result["status"] == "negated"

    def test_had_no_does_not_trigger_past_tense(self):
        # "Patient had no fever" — negation construction; TAM must not fire
        result_neg  = fuse("Patient has no fever")
        result_past = fuse("Patient had no fever")
        # Both should be negated
        assert result_neg["status"]  == "negated"
        assert result_past["status"] == "negated"

    def test_strong_resolved_cue_not_flipped_by_epistemic(self):
        # "history of" (weight 0.95) should still dominate even with "might"
        result = fuse("History of hypertension, which might recur")
        # resolved or ambiguous is acceptable; must not flip to ongoing
        assert result["status"] != "ongoing"

    def test_negated_deontic_boosts_negated_in_fusion(self):
        result = fuse("Active infection should not be present")
        # negated_deontic pulls toward negated; posterior["negated"] should
        # be the largest or second-largest mass
        posterior = result["posterior"]
        assert posterior["negated"] > posterior["resolved"]

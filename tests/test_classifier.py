"""
Classifier tests — organised from simple to tricky.

The test suite deliberately includes cases that *break* the original
first-match substring approach, demonstrating the value of each improvement:

  ABBREV  — abbreviation normalization
  SCOPE   — compound/NegEx negation scope ("has no", "no active")
  PSEUDO  — pseudo-negation masking ("no longer", "not improving")
  TEMPORAL — past/present temporal signals
  CONFLICT — phrases with competing signals
"""

import pytest
from src.classifier import classify_condition_status


# ---------------------------------------------------------------------------
# 1. Original baseline cases (must still pass)
# ---------------------------------------------------------------------------

def test_ongoing_simple():
    assert classify_condition_status("Asthma better today")["status"] == "ongoing"

def test_resolved_explicit():
    assert classify_condition_status("Fever has resolved")["status"] == "resolved"

def test_negated_denies():
    assert classify_condition_status("Patient denies chest pain")["status"] == "negated"

def test_ambiguous_possible():
    assert classify_condition_status("Possible pneumonia")["status"] == "ambiguous"

def test_ongoing_default():
    assert classify_condition_status("Patient has diabetes")["status"] == "ongoing"

def test_resolved_history_of():
    assert classify_condition_status("History of asthma")["status"] == "resolved"

def test_negated_no_evidence():
    assert classify_condition_status("No evidence of pneumonia")["status"] == "negated"

def test_ongoing_persistent():
    assert classify_condition_status("Persistent cough for 2 weeks")["status"] == "ongoing"

def test_ongoing_controlled():
    assert classify_condition_status("Seizures controlled on medication")["status"] == "ongoing"

def test_ambiguous_rule_out():
    assert classify_condition_status("Rule out sepsis")["status"] == "ambiguous"


# ---------------------------------------------------------------------------
# 2. ABBREV — abbreviation normalization (broke original system entirely)
# ---------------------------------------------------------------------------

def test_abbrev_h_o():
    """h/o diabetes → history of diabetes → resolved"""
    assert classify_condition_status("h/o diabetes")["status"] == "resolved"

def test_abbrev_s_p():
    """s/p appendectomy → status post → resolved"""
    assert classify_condition_status("s/p appendectomy")["status"] == "resolved"

def test_abbrev_c_o():
    """c/o chest pain → complains of chest pain → ongoing"""
    assert classify_condition_status("c/o chest pain")["status"] == "ongoing"

def test_abbrev_negative_result():
    """Fever -ve → negative for fever → negated"""
    assert classify_condition_status("Fever -ve")["status"] == "negated"

def test_abbrev_pmh():
    """PMH: type 2 diabetes → past medical history → resolved"""
    assert classify_condition_status("PMH: type 2 diabetes")["status"] == "resolved"

def test_abbrev_htn():
    """HTN well-controlled → hypertension well-controlled → ongoing"""
    assert classify_condition_status("HTN well-controlled")["status"] == "ongoing"


# ---------------------------------------------------------------------------
# 3. SCOPE — compound negation cues handle "verb + no" patterns
# ---------------------------------------------------------------------------

def test_scope_patient_has_no():
    """'Patient has no fever' — 'patient has' (ongoing) must NOT beat 'has no' (negated)"""
    assert classify_condition_status("Patient has no fever")["status"] == "negated"

def test_scope_no_active():
    """'No active infection' — 'active' (ongoing) must NOT beat 'no active' (negated)"""
    assert classify_condition_status("No active infection")["status"] == "negated"

def test_scope_shows_no():
    assert classify_condition_status("Imaging shows no evidence of fracture")["status"] == "negated"

def test_scope_with_no():
    assert classify_condition_status("Patient presents with no signs of sepsis")["status"] == "negated"


# ---------------------------------------------------------------------------
# 4. PSEUDO — pseudo-negation masking
# ---------------------------------------------------------------------------

def test_pseudo_no_longer_is_resolved_not_negated():
    """'No longer has headache' — 'no longer' must be resolved, not negated by bare 'no'"""
    assert classify_condition_status("No longer has headache")["status"] == "resolved"

def test_pseudo_not_improving_is_ongoing():
    """'Not improving on current regimen' — condition persists; should be ongoing"""
    assert classify_condition_status("Hypertension not improving on current regimen")["status"] == "ongoing"

def test_pseudo_no_improvement_is_ongoing():
    """'No improvement noted' — condition still present"""
    assert classify_condition_status("Cough, no improvement noted")["status"] == "ongoing"

def test_pseudo_no_change_is_ongoing():
    assert classify_condition_status("Diabetes, no change in status")["status"] == "ongoing"


# ---------------------------------------------------------------------------
# 5. TEMPORAL — past / present signals
# ---------------------------------------------------------------------------

def test_temporal_years_ago_resolved():
    """'DM diagnosed 3 years ago' — temporal past nudges toward resolved"""
    assert classify_condition_status("DM diagnosed 3 years ago")["status"] == "resolved"

def test_temporal_since_this_morning_ongoing():
    assert classify_condition_status("Chest pain since this morning")["status"] == "ongoing"

def test_temporal_recovered_last_month():
    assert classify_condition_status("Recovered from pneumonia last month")["status"] == "resolved"

def test_temporal_acute_onset():
    assert classify_condition_status("Acute onset chest pain")["status"] == "ongoing"


# ---------------------------------------------------------------------------
# 6. CONFLICT — phrases with competing signals (history of + currently)
# ---------------------------------------------------------------------------

def test_conflict_history_currently_worsening():
    """'History of asthma, currently worsening' — current worsening should win"""
    assert classify_condition_status("History of asthma, currently worsening")["status"] == "ongoing"

def test_conflict_prior_mi_presenting_chest_pain():
    """Prior MI + presenting chest pain — presenting wins for the current complaint"""
    result = classify_condition_status("Prior MI, presenting with chest pain")
    assert result["status"] == "ongoing"


# ---------------------------------------------------------------------------
# 7. Confidence and signal structure
# ---------------------------------------------------------------------------

def test_confidence_returned():
    result = classify_condition_status("Persistent cough")
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

def test_signals_dict_present():
    result = classify_condition_status("Possible pneumonia")
    assert "signals" in result
    for key in ("negated", "ambiguous", "resolved", "ongoing", "temporal"):
        assert key in result["signals"]

def test_empty_input():
    result = classify_condition_status("")
    assert result["status"] == "ambiguous"
    assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# 8. New ongoing cues — "just have", "now have", "still have"
# ---------------------------------------------------------------------------

def test_just_have_is_ongoing():
    """'Now, I just have cough' — 'just have' should score as ongoing."""
    result = classify_condition_status("Now, I just have cough")
    assert result["status"] == "ongoing"

def test_now_have_is_ongoing():
    result = classify_condition_status("Now I have nasal allergies")
    assert result["status"] == "ongoing"

def test_still_have_is_ongoing():
    result = classify_condition_status("I still have headache")
    assert result["status"] == "ongoing"

def test_currently_have_is_ongoing():
    result = classify_condition_status("I currently have asthma")
    assert result["status"] == "ongoing"


# ---------------------------------------------------------------------------
# 9. Clause override — "just have" raises final-clause confidence above 0.65
# ---------------------------------------------------------------------------

def test_clause_override_just_have_beats_resolved():
    """
    'I had cold. It got over. Now, I just have cough' —
    the final clause should override the resolved signal from 'got over'.
    """
    result = classify_condition_status(
        "I had cold. It got over. Now, I just have cough"
    )
    assert result["status"] == "ongoing"

def test_clause_override_now_have_beats_healed():
    result = classify_condition_status(
        "I had cold which got healed. Now, I just have nasal allergies"
    )
    assert result["status"] == "ongoing"

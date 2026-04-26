from src.classifier import classify_condition_status


def test_ongoing_condition():
    result = classify_condition_status("Asthma better today")
    assert result["status"] == "ongoing"


def test_resolved_condition():
    result = classify_condition_status("Fever has resolved")
    assert result["status"] == "resolved"


def test_negated_condition():
    result = classify_condition_status("Patient denies chest pain")
    assert result["status"] == "negated"


def test_ambiguous_condition():
    result = classify_condition_status("Possible pneumonia")
    assert result["status"] == "ambiguous"


def test_default_ongoing():
    result = classify_condition_status("Patient has diabetes")
    assert result["status"] == "ongoing"
"""
Generate a large labelled phrase dataset for Platt-scaling calibration.

Produces data/calibration_phrases.csv with columns:
    text, gold_status

Run with:
    python data/generate_calibration_dataset.py
"""

import csv
import itertools
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Conditions drawn from the NER vocabulary
# ---------------------------------------------------------------------------
CONDITIONS = [
    "hypertension", "diabetes", "asthma", "pneumonia", "chest pain",
    "fever", "cough", "headache", "migraine", "anxiety", "depression",
    "seizures", "sepsis", "pulmonary embolism", "deep vein thrombosis",
    "coronary artery disease", "atrial fibrillation", "heart failure",
    "stroke", "anemia", "osteoporosis", "rheumatoid arthritis",
    "shortness of breath", "fatigue", "edema", "nausea", "vertigo",
    "insomnia", "obesity", "gout",
]

# ---------------------------------------------------------------------------
# Templates — each {condition} slot is filled with every condition above.
# Gold label is given by the template category.
# ---------------------------------------------------------------------------

RESOLVED_TEMPLATES = [
    "History of {c}",
    "h/o {c}",
    "Past history of {c}",
    "Prior {c}",
    "{c} in the past",
    "{c} resolved",
    "{c} has resolved",
    "{c} fully resolved",
    "{c} resolved after treatment",
    "Previous episode of {c}",
    "{c} 3 years ago",
    "{c} 2 months ago",
    "{c} last year",
    "Recovered from {c}",
    "{c} in remission",
    "Had {c} last year",
    "{c} no longer present",
    "No longer has {c}",
    "{c} treated successfully",
    "Former {c}",
    "{c} previously diagnosed",
    "s/p treatment for {c}",
    "Status post {c}",
    "{c} was treated",
    "{c} cleared 6 months ago",
]

ONGOING_TEMPLATES = [
    "Patient has {c}",
    "{c} stable",
    "{c} is stable",
    "{c} well-controlled",
    "Persistent {c}",
    "Chronic {c}",
    "{c} ongoing",
    "{c} active",
    "{c} currently present",
    "Currently has {c}",
    "Presenting with {c}",
    "Worsening {c}",
    "{c} worsening",
    "Active {c}",
    "{c} controlled on medication",
    "Stable {c}",
    "Managed {c}",
    "{c} improving",
    "Acute {c}",
    "{c} for the past 3 days",
    "{c} since this morning",
    "{c} not improving on current regimen",
    "{c} remains present",
    "{c} continues",
    "Complains of {c}",
]

NEGATED_TEMPLATES = [
    "No {c}",
    "No evidence of {c}",
    "Patient denies {c}",
    "Denies {c}",
    "{c} negative",
    "Negative for {c}",
    "No active {c}",
    "Absence of {c}",
    "Without {c}",
    "{c} not present",
    "No history of {c}",
    "No signs of {c}",
    "Patient has no {c}",
    "{c} ruled out",
    "{c} absent",
    "No known {c}",
    "{c} -ve",
    "Imaging shows no evidence of {c}",
    "Does not have {c}",
    "No current {c}",
    "Free of {c}",
    "{c} not detected",
    "No documented {c}",
    "Denies any history of {c}",
    "No complaints of {c}",
]

AMBIGUOUS_TEMPLATES = [
    "Possible {c}",
    "Probable {c}",
    "Suspected {c}",
    "Rule out {c}",
    "Concern for {c}",
    "Cannot rule out {c}",
    "Questionable {c}",
    "May have {c}",
    "Possible history of {c}",
    "Query {c}",
    "? {c}",
    "Possible new {c}",
    "Differential includes {c}",
    "Working diagnosis of {c}",
    "Considering {c}",
    "Atypical presentation of {c}",
    "Symptoms consistent with {c}",
    "Could represent {c}",
    "Unlikely but possible {c}",
    "To rule out {c}",
]

# ---------------------------------------------------------------------------
# Generate rows
# ---------------------------------------------------------------------------

def expand_templates(templates, label, conditions):
    rows = []
    for tmpl, cond in itertools.product(templates, conditions):
        rows.append({"text": tmpl.format(c=cond), "gold_status": label})
    return rows


def generate():
    rows = []
    rows += expand_templates(RESOLVED_TEMPLATES,  "resolved",  CONDITIONS)
    rows += expand_templates(ONGOING_TEMPLATES,   "ongoing",   CONDITIONS)
    rows += expand_templates(NEGATED_TEMPLATES,   "negated",   CONDITIONS)
    rows += expand_templates(AMBIGUOUS_TEMPLATES, "ambiguous", CONDITIONS)

    random.shuffle(rows)

    out = Path(__file__).parent / "calibration_phrases.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "gold_status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} phrases to {out}")
    return out


if __name__ == "__main__":
    generate()

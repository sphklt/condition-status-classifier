"""
Clinical Named Entity Recognition (NER).

Two paths — selected automatically at runtime:

  Primary  : SciSpaCy with en_ner_bc5cdr_md
             Trained on the BC5CDR corpus; returns DISEASE entities.
             Install: pip install scispacy
                      pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

  Fallback : Vocabulary-based matcher using a curated list of ~80 common
             clinical conditions and symptoms.  Works out of the box with
             zero additional dependencies.  Less recall than SciSpaCy but
             demonstrates the correct pipeline architecture.

Call extract_entities(text) — it automatically picks the best available path.
Call active_ner_method() to see which one is in use.
"""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared data type
# ---------------------------------------------------------------------------

@dataclass
class MedicalEntity:
    text: str
    start: int   # char offset within the input text
    end: int
    label: str   # "DISEASE" (scispacy) or "CONDITION" (fallback)


# ---------------------------------------------------------------------------
# SciSpaCy path
# ---------------------------------------------------------------------------

_nlp = None
_SCISPACY_AVAILABLE = None   # None = not yet tried
_MODEL_NAME = "en_ner_bc5cdr_md"


def _try_load_scispacy() -> bool:
    global _nlp, _SCISPACY_AVAILABLE
    if _SCISPACY_AVAILABLE is not None:
        return _SCISPACY_AVAILABLE
    try:
        import spacy  # noqa: F401
        _nlp = spacy.load(_MODEL_NAME)
        _SCISPACY_AVAILABLE = True
    except Exception:
        _SCISPACY_AVAILABLE = False
    return _SCISPACY_AVAILABLE


def _extract_scispacy(text: str) -> list[MedicalEntity]:
    doc = _nlp(text)
    entities = [
        MedicalEntity(ent.text, ent.start_char, ent.end_char, ent.label_)
        for ent in doc.ents
        if ent.label_ in ("DISEASE", "CHEMICAL")
    ]
    return _apply_supplemental(text, entities)


# ---------------------------------------------------------------------------
# Vocabulary fallback
# ---------------------------------------------------------------------------

# Curated list covering the most common conditions in clinical notes.
# Longer / more specific phrases come first so the regex alternation
# prefers them over shorter sub-strings (e.g. "heart failure" before "failure").
_VOCAB: list[str] = [
    # Cardiovascular
    "congestive heart failure", "heart failure", "atrial fibrillation",
    "coronary artery disease", "myocardial infarction", "heart attack",
    "deep vein thrombosis", "pulmonary embolism", "peripheral artery disease",
    "hypertension", "hypotension", "angina", "arrhythmia", "stroke", "tia",
    "aortic stenosis", "mitral regurgitation",
    # Respiratory
    "chronic obstructive pulmonary disease", "pulmonary fibrosis",
    "shortness of breath", "sleep apnea", "respiratory failure",
    "asthma", "pneumonia", "bronchitis", "pleuritis", "dyspnea",
    "wheezing", "cough", "hemoptysis",
    # Endocrine / metabolic
    "type 2 diabetes mellitus", "type 1 diabetes mellitus",
    "diabetes mellitus", "diabetes",
    "hypothyroidism", "hyperthyroidism", "hyperlipidemia", "obesity",
    "metabolic syndrome", "gout",
    # Gastrointestinal
    "inflammatory bowel disease", "crohn's disease", "ulcerative colitis",
    "peptic ulcer disease", "peptic ulcer", "gastroesophageal reflux disease",
    "irritable bowel syndrome", "hepatitis", "cirrhosis", "pancreatitis",
    "appendicitis", "cholecystitis", "diverticulitis",
    "gerd", "nausea", "vomiting", "abdominal pain", "diarrhea", "constipation",
    # Musculoskeletal
    "rheumatoid arthritis", "osteoarthritis", "ankylosing spondylitis",
    "systemic lupus erythematosus", "fibromyalgia", "osteoporosis",
    "arthritis", "back pain", "joint pain", "fracture",
    # Neurological
    "parkinson's disease", "alzheimer's disease", "multiple sclerosis",
    "amyotrophic lateral sclerosis", "peripheral neuropathy",
    "migraine", "headache", "seizure", "epilepsy", "dementia", "neuropathy",
    "syncope", "vertigo",
    # Infectious
    "urinary tract infection", "upper respiratory infection",
    "common cold",
    "sepsis", "cellulitis", "pneumonia", "meningitis",
    "covid-19", "influenza", "cold", "flu", "fever", "infection",
    # Oncology
    "non-hodgkin lymphoma", "hodgkin lymphoma", "chronic lymphocytic leukemia",
    "breast cancer", "lung cancer", "colon cancer", "prostate cancer",
    "carcinoma", "lymphoma", "leukemia", "malignancy", "tumor",
    "cancer", "metastasis",
    # Renal / urological
    "chronic kidney disease", "acute kidney injury", "renal failure",
    "kidney disease", "nephrotic syndrome", "urinary incontinence",
    # Mental health
    "major depressive disorder", "bipolar disorder", "post-traumatic stress disorder",
    "generalized anxiety disorder", "schizophrenia",
    "depression", "anxiety", "ptsd", "insomnia",
    # Haematological
    "deep vein thrombosis", "anemia", "thrombocytopenia", "neutropenia",
    # Symptoms (broad)
    "chest pain", "chest tightness", "palpitations", "weight loss",
    "weight gain", "fatigue", "edema", "peripheral edema",
]

_FALLBACK_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(_VOCAB, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Supplemental vocabulary — colloquial terms SciSpaCy's BC5CDR model misses.
# Applied after SciSpaCy extraction; skipped if the span is already covered.
# ---------------------------------------------------------------------------
_SUPPLEMENTAL: list[str] = [
    "common cold", "cold", "flu",
    "nasal allergies", "allergies",
]

_SUPPLEMENTAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(_SUPPLEMENTAL, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _apply_supplemental(text: str, existing: list[MedicalEntity]) -> list[MedicalEntity]:
    """Add supplemental matches for spans not already covered by *existing* entities."""
    extras = []
    for m in _SUPPLEMENTAL_RE.finditer(text):
        overlaps = any(m.start() < e.end and m.end() > e.start for e in existing)
        if not overlaps:
            extras.append(MedicalEntity(m.group(), m.start(), m.end(), "CONDITION"))
    return existing + extras


def _extract_vocab(text: str) -> list[MedicalEntity]:
    return [
        MedicalEntity(m.group(), m.start(), m.end(), "CONDITION")
        for m in _FALLBACK_RE.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> list[MedicalEntity]:
    """
    Extract medical condition entities from *text*.
    Automatically uses SciSpaCy if installed; falls back to vocabulary matching.
    """
    if _try_load_scispacy():
        return _extract_scispacy(text)
    return _extract_vocab(text)


def active_ner_method() -> str:
    """Returns which NER method will be used."""
    return "scispacy" if _try_load_scispacy() else "vocabulary"

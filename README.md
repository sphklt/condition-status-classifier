# Clinical Condition Status Classifier

A clinical NLP system that classifies whether a medical condition mentioned in clinical text is **ongoing**, **resolved**, **negated**, or **ambiguous**.

The system operates at two levels:

- **Phrase level** — classifies a single short clinical phrase with confidence score and reasoning
- **Note level** — processes a full clinical note end-to-end: detects sections, extracts conditions via NER, and classifies each condition in its sentence context

Both levels are fully rule-based and explainable. Every prediction returns the matched signal, the reason, and a confidence score.

---

## Status Labels

| Label | Meaning | Examples |
|---|---|---|
| `ongoing` | Condition is active, persistent, stable, controlled, or currently changing | `"Persistent cough for 2 weeks"`, `"Diabetes is stable"`, `"Seizures controlled on medication"` |
| `resolved` | Condition is historical, closed, or no longer active | `"History of asthma"`, `"Fever has resolved"`, `"s/p appendectomy"` |
| `negated` | Condition is explicitly denied or absent | `"Patient denies chest pain"`, `"No evidence of pneumonia"`, `"Fever -ve"` |
| `ambiguous` | Condition is uncertain, suspected, or unconfirmed | `"Possible pneumonia"`, `"Rule out sepsis"`, `"Concern for pulmonary embolism"` |

**Important nuance — "better" is ongoing, not resolved:**

```text
Asthma better today  →  ongoing
```

"Better" means the condition has improved but is still present. Words like `stable`, `controlled`, `improving`, and `better` all signal an active condition being managed — not a resolved one.

---

## What Makes This System Non-Trivial

Most clinical condition classifiers use simple keyword lookup — scan for a word, return a label. This system addresses four failure modes that keyword lookup cannot handle.

### 1. Pseudo-negation filtering (NegEx-inspired)

Some phrases contain negation words that do **not** deny a condition. A keyword system classifies all of these as `negated`. This system does not.

| Phrase | Naive system | This system | Why |
|---|---|---|---|
| `No longer has headache` | `negated` (sees "no") | `resolved` | "no longer" = condition ended, not denied |
| `Not improving on current regimen` | `negated` (sees "not") | `ongoing` | condition still present, just not responding |
| `No improvement noted` | `negated` (sees "no") | `ongoing` | condition persists unchanged |
| `No change in diabetes status` | `negated` (sees "no") | `ongoing` | unchanged = still present |

These patterns are detected before scoring and their spans are masked so the negation cue cannot fire on them. This is directly inspired by the **NegEx algorithm** (Chapman et al., 2001), a foundational method in clinical NLP.

### 2. Adversative clause detection

When a sentence has two clauses separated by an adversative conjunction ("But", "However") or a period, the **final clause carries the definitive status**. A bag-of-words system averages all signals equally.

```text
"I had severe flu which I think is getting better now.
 But after a couple of days, it got completely over."
```

| System | Result | Why it fails |
|---|---|---|
| Keyword (original) | `ongoing` — sees "better", "now" | treats the whole sentence as one bag of words |
| This system | `resolved` — reads "completely over" | splits on the period, classifies the final clause separately |

The final clause rule reflects a real property of clinical language: clinicians and patients often state the *current* status last, after describing how things were before.

### 3. Temporal signal detection

Time expressions are a strong, underused classification signal. The system detects past and present temporal patterns independently from keyword cues and uses them to boost the relevant category score.

| Phrase | Temporal signal | Effect |
|---|---|---|
| `"DM diagnosed 3 years ago"` | past (`"3 years ago"`) | boosts resolved score |
| `"Chest pain since this morning"` | present (`"this morning"`) | boosts ongoing score |
| `"Was treated for pneumonia last year"` | past (`"last year"` + `"was treated"`) | boosts resolved score |
| `"Acute onset shortness of breath"` | present (`"acute"`) | boosts ongoing score |

This handles cases where no explicit resolved/ongoing keyword exists — the time expression alone is sufficient evidence.

### 4. Sentence-boundary-aware context windows

When classifying an entity within a longer section of text, the context must not bleed across sentence boundaries.

**The problem with fixed character windows:**

```text
"No fever.  Patient has diabetes."
 ^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
 sentence 1   sentence 2
```

A 120-character window around "diabetes" would include "No fever", causing the negation cue "No" to pollute the diabetes classification.

**This system's approach:**
- Splits section text into sentences using a clinical sentence boundary detector
- Protects abbreviations (`Dr.`, `b.i.d.`, `e.g.`) so they don't create false boundaries
- Deliberately does NOT protect units (`mg`, `kg`) — when followed by a capital word, they reliably mark real sentence boundaries
- Each entity is classified using only the sentence it appears in

Result: "No fever" never contaminates "Patient has diabetes."

---

## Architecture

The system operates as a two-level pipeline.

### Level 1 — Phrase Classifier

```
Raw phrase
    │
    ▼
Abbreviation normalizer     h/o → history of, -ve → negative for, s/p → status post
    │
    ▼
Pseudo-negation masking     "no longer", "not improving" → masked before cue matching
    │
    ▼
Multi-signal cue matching   word-boundary regex, weighted scores per category
    │
    ▼
Temporal signal detection   past/present time expressions boost resolved/ongoing
    │
    ▼
Adversative clause check    "But…" / "However…" / "." → classify final clause
    │
    ▼
Conflict detection          competing signals → reduced confidence
    │
    ▼
{status, confidence, cue, reason, signals}
```

### Level 2 — Full Note Pipeline

```
Raw clinical note
    │
    ▼
Section detector            PMH → resolved prior | HPI → ongoing prior | Assessment → ongoing prior
    │
    ▼
(per section)
    │
    ▼
Abbreviation normalizer
    │
    ▼
Sentence splitter           splits section into sentences with char offsets
    │
    ▼
NER                         SciSpaCy (if installed) or vocabulary-based fallback
    │
    ▼
Context extraction          finds the sentence containing each entity (not a fixed char window)
    │
    ▼
Phrase classifier           classifies each entity in its sentence context
    │
    ▼
Section prior override      low-confidence result? section prior takes over
    │
    ▼
Deduplication               first mention of each condition wins
    │
    ▼
[{condition, status, confidence, section, reason}, ...]
```

---

## Module Reference

| Module | Role |
|---|---|
| `src/normalizer.py` | Expands clinical abbreviations before any matching |
| `src/rules.py` | Weighted cue lists (100+ entries) + pseudo-negation registry |
| `src/temporal.py` | Detects past/present temporal expressions |
| `src/classifier.py` | Orchestrates phrase-level classification |
| `src/section_detector.py` | Splits clinical notes into labeled sections |
| `src/ner.py` | Named entity extraction (SciSpaCy primary, vocabulary fallback) |
| `src/sentence_splitter.py` | Clinical sentence boundary detection |
| `src/pipeline.py` | Orchestrates the full note-level pipeline |
| `src/utils.py` | Dataset evaluation helper |

---

## Project Structure

```text
condition-status-classifier/
│
├── data/
│   └── clinical_phrases.csv          39-phrase labelled dataset (hard cases included)
│
├── src/
│   ├── __init__.py
│   ├── normalizer.py                 abbreviation expansion
│   ├── rules.py                      weighted cues + pseudo-negation patterns
│   ├── temporal.py                   temporal signal detection
│   ├── classifier.py                 phrase-level classifier
│   ├── section_detector.py           note section splitter
│   ├── ner.py                        NER (SciSpaCy / vocabulary fallback)
│   ├── sentence_splitter.py          sentence boundary detection
│   ├── pipeline.py                   full note pipeline
│   └── utils.py                      dataset evaluation
│
├── tests/
│   ├── test_classifier.py            33 phrase-level tests
│   └── test_pipeline.py              33 pipeline tests (section, NER, pipeline, sentence splitter)
│
├── app.py                            Streamlit app (phrase + note + evaluation tabs)
├── main.py                           CLI evaluation entrypoint
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Optional — SciSpaCy NER (improves entity recall on real notes):**

> **Python version note:** SciSpaCy's dependencies do not have pre-built wheels for Python 3.13. Use Python 3.11 for the virtual environment.

```bash
# Create venv with Python 3.11 if needed (pyenv example)
~/.pyenv/versions/3.11.8/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Then install SciSpaCy and the BC5CDR disease NER model
pip install scispacy==0.5.4
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

The system detects SciSpaCy automatically. If not installed, it falls back to a vocabulary-based NER covering ~80 common clinical conditions with no setup required.

---

## Usage

### CLI — evaluate the labelled dataset

```bash
python main.py
```

```text
                                           text gold_status predicted_status        matched_cue  confidence  is_correct
                         The patient has asthma     ongoing          ongoing        patient has        0.78        True
                            Asthma better today     ongoing          ongoing             better        0.82        True
                Asthma resolved after treatment    resolved         resolved     resolved after        1.00        True
                       No evidence of pneumonia     negated          negated     no evidence of        1.00        True
                         No longer has headache    resolved         resolved      no longer has        1.00        True
                                   h/o diabetes    resolved         resolved         history of        0.95        True
                                 c/o chest pain     ongoing          ongoing       complains of        1.00        True
                                      Fever -ve     negated          negated       negative for        1.00        True
         History of asthma, currently worsening     ongoing          ongoing          worsening        0.80        True

Accuracy: 97.44 %  (38/39 correct)
```

### CLI — process a full clinical note

```python
from src.pipeline import process_note, format_results

note = """
History of Present Illness:
67-year-old female presenting with worsening dyspnea.
She reports fatigue for 3 days. Denies chest pain or fever.

Past Medical History:
Hypertension, type 2 diabetes mellitus (diagnosed 5 years ago),
h/o pneumonia (resolved last year), atrial fibrillation controlled on medication.

Assessment:
Possible heart failure exacerbation. Rule out pulmonary embolism.
Hypertension well-controlled. Diabetes stable.
"""

print(format_results(process_note(note)))
```

```text
NER method : vocabulary
Sections   : history_of_present_illness, past_medical_history, assessment

CONDITION                           STATUS         CONF  SECTION
----------------------------------------------------------------
dyspnea                             ongoing        100%  history_of_present_illness
shortness of breath                 ongoing        100%  history_of_present_illness
chest pain                          negated        100%  history_of_present_illness
fever                               negated        100%  history_of_present_illness
Hypertension                        resolved       100%  past_medical_history
type 2 diabetes mellitus            resolved       100%  past_medical_history
pneumonia                           resolved       100%  past_medical_history
atrial fibrillation                 resolved       100%  past_medical_history
heart failure                       ambiguous       76%  assessment
pulmonary embolism                  ambiguous       95%  assessment
Diabetes                            ongoing         80%  assessment
```

### Run tests

```bash
pytest
```

```text
66 passed in 0.04s
```

### Run the Streamlit app

```bash
streamlit run app.py
```

The app has three tabs:

| Tab | What it does |
|---|---|
| Single Phrase | Classify a phrase; shows confidence, signal scores, abbreviations expanded, clause used |
| Full Clinical Note | Paste a clinical note; runs the full pipeline and returns a colour-coded condition table |
| Evaluate Dataset | Runs the classifier over the labelled CSV and reports accuracy |

---

## Module Details

### `src/normalizer.py` — Abbreviation expansion

Real clinical notes are dense with shorthand. The normalizer expands abbreviations before any cue matching so the classifier sees full English phrases.

| Input | Expanded | Effect on classification |
|---|---|---|
| `h/o diabetes` | `history of diabetes` | `history of` fires as resolved cue |
| `c/o chest pain` | `complains of chest pain` | `complains of` fires as ongoing cue |
| `s/p appendectomy` | `status post appendectomy` | `status post` fires as resolved cue |
| `Fever -ve` | `Fever negative for` | `negative for` fires as negation cue |
| `HTN well-controlled` | `hypertension well-controlled` | `well-controlled` fires as ongoing cue |
| `PMH: diabetes` | `past medical history: diabetes` | `past medical history` fires as resolved cue |

### `src/rules.py` — Weighted cues

Each cue is a `(phrase, weight)` tuple. Weight reflects reliability — a 4-word phrase like `"no evidence of"` (weight 1.0) is a much stronger signal than bare `"no"` (weight 0.6).

Cues are grouped into four categories, each with ~25 entries ordered from most specific to least specific:

```python
NEGATION_CUES = [
    ("patient denies any", 1.0),
    ("no evidence of",     1.0),
    ("has no",             0.95),   # compound — handles "patient has no fever"
    ("no active",          0.90),   # compound — handles "no active infection"
    ("without",            0.70),
    ("no",                 0.60),   # weakest — single word, many false positives
    ...
]
```

**Compound negation cues** are the key addition over a simple keyword list. Without them:

```text
"Patient has no fever"
```

A naive system sees `"patient has"` (ongoing, strong) and `"no"` (negated, weak) and picks **ongoing**. This system includes `"has no"` (negated, weight 0.95) as a compound cue, which outscores `"patient has"` (ongoing, weight 0.70).

**Pseudo-negation patterns** (in `PSEUDO_NEGATION_PATTERNS`) are phrases that syntactically look like negations but semantically are not:

```python
PSEUDO_NEGATION_PATTERNS = [
    r"\bnot improving\b",      # condition persists — ongoing, not negated
    r"\bno improvement\b",     # condition persists — ongoing, not negated
    r"\bnot only\b",           # additive construction
    r"\bnot fully\b",          # partial state — not a full negation
    r"\bno change\b",          # condition unchanged — ongoing, not negated
    ...
]
```

### `src/temporal.py` — Temporal signal detection

Temporal expressions give strong evidence about condition status independently from keyword cues. The module detects two signal types using regex:

**Past signals** → boost `resolved` score:
- Numeric: `"3 years ago"`, `"2 months ago"`
- Relative: `"last year"`, `"last month"`
- Absolute: `"in 2019"`, `"in 2020"`
- Adverbial: `"previously"`, `"formerly"`, `"historically"`
- Verb tense: `"was diagnosed with"`, `"was treated for"`, `"had an episode of"`

**Present signals** → boost `ongoing` score:
- Adverbial: `"currently"`, `"now"`, `"today"`
- Relative: `"this week"`, `"this morning"`, `"since yesterday"`
- Duration: `"for the past 3 days"`, `"over the last week"`
- Presentation: `"presents with"`, `"complains of"`
- Clinical: `"acute"`, `"acutely"`

### `src/classifier.py` — Phrase-level classification

**Scoring approach** — all matching cues contribute, not just the first match:

```
category_score = max_weight_found + multi_cue_bonus (capped at 1.0)
```

This means "History of asthma, currently worsening" is correctly classified as `ongoing`:

```
resolved: "history of" (0.95)  →  score 0.95
ongoing:  "currently" (0.90) + "worsening" (0.95)  →  score 0.95 + 0.08 bonus = 1.0

ongoing wins
```

The original first-match priority system returned `resolved` for this phrase.

**Adversative clause detection:**

When text contains multiple sentences or an adversative conjunction, the pipeline:
1. Tries to split on a period + capital letter
2. Falls back to splitting on `"but"`, `"however"`, `"although"` etc.
3. Classifies the final clause separately
4. If that clause has confidence ≥ 0.65, uses it (with a 8% confidence discount for ignoring earlier context)

```text
"Getting better now. But it got completely over."
 ──────────────────  ──────────────────────────
  clause 1: ongoing   clause 2: resolved (wins)
```

### `src/section_detector.py` — Note section detection

Clinical note sections carry strong status priors:

| Section | Status prior | Rationale |
|---|---|---|
| Past Medical History | `resolved` | Conditions listed here were active in the past |
| Past Surgical History | `resolved` | Procedures are historical events |
| Chief Complaint | `ongoing` | The reason for the visit — active by definition |
| History of Present Illness | `ongoing` | Current presentation |
| Assessment | `ongoing` | Active working diagnoses |
| Family History | *(skipped)* | These are family members' conditions, not the patient's |

Headers are detected via regex at the start of a line, accepting both full names (`"Past Medical History:"`) and abbreviations (`"PMH:"`).

When the phrase classifier returns low confidence (< 0.55) and the section has a prior, the prior overrides. This handles bare condition lists like:

```text
Past Medical History:
Hypertension, diabetes, atrial fibrillation.
```

No cue phrase appears with any of those conditions, but the section header makes the correct answer obvious: all three are `resolved`.

### `src/sentence_splitter.py` — Sentence boundary detection

Splits text into sentences with character offsets, solving the context-pollution problem.

**Abbreviation protection:** periods in known abbreviations are temporarily replaced with a null-byte guard so the boundary regex ignores them.

```text
"Dr. Smith examined the patient. He has hypertension."
 ─── protected                   ─── real boundary
```

**Key design decision — units are NOT protected:**

`"mg"`, `"kg"`, `"ml"` are deliberately excluded from the protection list. When a unit abbreviation is followed by a capital word, it is almost always a real sentence boundary:

```text
"Patient takes 500 mg. She has diabetes."
              ──────── real boundary → 2 sentences ✓
```

The only risk would be `"5-yr. history"` — but here `"history"` starts with lowercase, so the boundary regex (which requires a capital letter after the period) correctly ignores it.

### `src/ner.py` — Named entity recognition

Two paths, selected automatically:

**Primary — SciSpaCy `en_ner_bc5cdr_md`:**
- Trained on the BC5CDR corpus (disease and chemical entities)
- Returns `DISEASE` entities with character offsets
- Install: see Installation section above

**Fallback — vocabulary matching:**
- ~80 common clinical conditions compiled into a single alternation regex
- Ordered longest-to-shortest so `"congestive heart failure"` matches before `"heart failure"` before `"failure"`
- Works with zero setup — used automatically when SciSpaCy is not installed

```python
from src.ner import extract_entities, active_ner_method
print(active_ner_method())          # "scispacy" or "vocabulary"
entities = extract_entities("Patient has hypertension and no evidence of diabetes.")
# → [MedicalEntity("hypertension", 12, 24, "CONDITION"),
#    MedicalEntity("diabetes", 50, 58, "CONDITION")]
```

---

## Dataset

`data/clinical_phrases.csv` contains 39 labelled phrases, expanded from the original 15 to include harder cases that expose common failure modes:

| Category | Examples |
|---|---|
| Abbreviation cases | `"h/o diabetes"`, `"c/o chest pain"`, `"Fever -ve"`, `"HTN well-controlled"` |
| Compound negation scope | `"Patient has no fever"`, `"No active infection"`, `"Imaging shows no evidence of fracture"` |
| Pseudo-negation | `"No longer has headache"`, `"Hypertension not improving on current regimen"` |
| Temporal signals | `"DM diagnosed 3 years ago"`, `"Chest pain since this morning"` |
| Conflicting signals | `"History of asthma, currently worsening"`, `"Prior MI, presenting with chest pain"` |

---

## Classifier Output Schema

Every classification returns a consistent dictionary:

```python
{
    "status":     "ongoing",           # ongoing | resolved | negated | ambiguous
    "confidence": 0.82,                # 0.0 – 1.0
    "cue":        "worsening",         # highest-weight matched phrase
    "reason":     "Ongoing/active cue found: 'worsening' | Temporal hint: present ('currently')",
    "signals": {
        "negated":          0.0,
        "ambiguous":        0.0,
        "resolved":         0.95,
        "ongoing":          1.0,
        "temporal":         "present",
        "pseudo_negations": [],
        "abbreviations":    ["htn → hypertension"],
        "clause_used":      "full"     # or "final_clause"
    }
}
```

---

## Design Decisions

### Why rule-based rather than ML?

Clinical NLP demands interpretability. Every prediction this system makes can be traced back to a specific cue, a temporal expression, or a section prior. A clinician or engineer auditing a prediction can see exactly why the label was assigned and correct the rules if they disagree.

An ML model offers higher recall at the cost of opacity. For a system whose predictions could inform clinical decisions, a transparent rule system with known failure modes is a safer starting point.

### Why weighted multi-signal scoring rather than priority order?

The original system used a fixed priority order: negated → ambiguous → resolved → ongoing. This fails when two signals appear in the same phrase:

```text
"History of asthma, currently worsening"
```

- Priority order: `resolved` wins (first cue found)
- Weighted scoring: `ongoing` wins (worsening 0.95 + currently 0.90 + multi-cue bonus > history of 0.95)

Weighted scoring is better because it reflects how much evidence supports each label, not just which label had a cue appear first.

### Why a two-level pipeline rather than a single classifier?

The phrase classifier is designed for short, focused input — a clause or sentence about one condition. Real clinical notes are long and mention many conditions in different contexts and sections.

The pipeline feeds the phrase classifier exactly what it needs: a single sentence from the appropriate section. This separation of concerns means the classifier does not need to change as the pipeline becomes more sophisticated (e.g. adding NER models, section-specific logic, coreference resolution).

---

## Known Limitations

| Limitation | What it means in practice |
|---|---|
| No syntactic parsing | Cannot resolve which noun a modifier attaches to — "Atrial fibrillation, previously in sinus rhythm" misclassifies because "previously" modifies the rhythm, not the AF |
| NER vocabulary ceiling | Rare conditions, specialist terms, and misspellings are missed by the fallback NER |
| Sentence-level only | No coreference — "The patient had a cough. It resolved." requires knowing "It" = "the cough" |
| Fixed section prior threshold | The 0.55 confidence threshold for section prior override is not empirically calibrated |
| No list-scope negation | "Denies fever, chills, or chest pain" — the negation is correctly found for each, but only coincidentally; there is no explicit list parser |
| Confidence is not calibrated | A confidence of 0.85 means "a strong cue was found", not "this prediction is correct 85% of the time" |

---

## What Comes Next

| Step | Effort | Impact |
|---|---|---|
| Sentence-boundary-aware context | Done | Eliminates cross-sentence signal pollution |
| Section detection | Done | Free confidence boost from note structure |
| Clinical NER (SciSpaCy) | Done | Unlocks real note processing |
| Dependency parsing | Medium | Fixes modifier-scope errors (requires spaCy) |
| Calibrated confidence | Medium | Enables reliable downstream thresholds |
| Annotated real-note evaluation | Medium | Shows true precision/recall on clinical data |
| Fine-tuned BERT / LLM | High | Handles novel phrasing, rare conditions, implicit context |

---

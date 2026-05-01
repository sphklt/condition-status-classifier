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

Most clinical condition classifiers use simple keyword lookup — scan for a word, return a label. This system addresses five failure modes that keyword lookup cannot handle.

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

### 5. Pronoun coreference within sections

Clinical notes often describe a condition in one sentence and update its status in the next using a pronoun. A sentence-level classifier misses this entirely.

```text
"The patient had a cough. It resolved."
 ─────────────────────── ────────────
  entity sentence: weak   pronoun sentence: strong resolved signal
```

| Approach | Result for "cough" | Why |
|---|---|---|
| Sentence-only | `ongoing` (35% confidence) | "had a cough" has no explicit status cue |
| With coreference | `resolved` (87% confidence) | "It resolved." is attributed to the most recent entity |

The coref step fires only when the entity's own sentence is uninformative (confidence < 0.65), ensuring it never overrides a confident existing classification (e.g., `"No evidence of pneumonia"` stays `negated` even if followed by `"It appears resolved."`). Coreference is scoped to the current section — a pronoun in Assessment cannot overwrite a PMH classification.

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
Dep parser refinement       negation/temporal scope via spaCy parse tree; corrects out-of-scope signals
    │
    ▼
Section prior override      low-confidence result? section prior takes over
    │
    ▼
Pronoun coreference         "It resolved." → updates most recent weak-confidence entity
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
| `src/classifier.py` | Orchestrates phrase-level classification; emits `calibrated_confidence` |
| `src/dep_parser.py` | spaCy dependency parsing: negation scope, list negation, temporal modifier scope |
| `src/section_detector.py` | Splits clinical notes into labeled sections |
| `src/ner.py` | Named entity extraction (SciSpaCy primary, vocabulary fallback) |
| `src/sentence_splitter.py` | Clinical sentence boundary detection |
| `src/coref.py` | Pronoun-to-entity coreference within sections |
| `src/pipeline.py` | Orchestrates the full note-level pipeline; wires dep parser refinement |
| `src/calibration.py` | Platt scaler (`calibrate()`) + reliability diagram + ECE |
| `src/baseline.py` | TF-IDF + logistic regression baseline for comparison with the rule system |
| `src/note_evaluator.py` | Precision/recall/F1 evaluation on annotated clinical notes |
| `src/utils.py` | Phrase-level dataset evaluation helper |

---

## Project Structure

```text
condition-status-classifier/
│
├── data/
│   ├── clinical_phrases.csv          127-phrase labelled dataset (hard cases + calibration eval set)
│   ├── annotated_notes.json          4 annotated clinical notes for pipeline P/R/F1 evaluation
│   ├── calibration.json              fitted Platt scaler parameters (a, b)
│   ├── calibration_phrases.csv       2,850 synthetic phrases used to fit the Platt scaler
│   └── generate_calibration_dataset.py  script that produced calibration_phrases.csv
│
├── src/
│   ├── __init__.py
│   ├── normalizer.py                 abbreviation expansion
│   ├── rules.py                      weighted cues + pseudo-negation patterns
│   ├── temporal.py                   temporal signal detection
│   ├── classifier.py                 phrase-level classifier (emits calibrated_confidence)
│   ├── dep_parser.py                 spaCy dep-tree: negation scope, list negation, temporal scope
│   ├── section_detector.py           note section splitter
│   ├── ner.py                        NER (SciSpaCy / vocabulary fallback)
│   ├── sentence_splitter.py          sentence boundary detection
│   ├── coref.py                      pronoun-to-entity coreference
│   ├── pipeline.py                   full note pipeline (dep parser refinement wired in)
│   ├── calibration.py                Platt scaler + reliability diagram + ECE
│   ├── baseline.py                   TF-IDF + logistic regression baseline
│   ├── note_evaluator.py             pipeline P/R/F1 evaluation on annotated notes
│   └── utils.py                      phrase-level dataset evaluation
│
├── experiments/
│   └── calibration_transfer.py       standalone reproducible calibration transfer experiment
│
├── tests/
│   ├── test_classifier.py            33 phrase-level tests
│   ├── test_pipeline.py              44 pipeline tests (section, NER, pipeline, sentence splitter)
│   ├── test_coref.py                 21 tests (coref unit, pipeline integration, calibration, evaluator)
│   └── test_dep_and_calibration.py   38 tests (dep parser, Platt calibration, calibration transfer)
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

The system detects SciSpaCy automatically. If not installed, it falls back to a vocabulary-based NER covering ~85 common clinical conditions (including colloquial terms like `"cold"` and `"flu"`) with no setup required.

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
110 passed in 5.34s
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
| Evaluate Dataset | Five sub-sections: (1) phrase accuracy over the 127-phrase CSV, (2) confidence reliability diagram + ECE, (3) ML baseline comparison (TF-IDF + LR vs rule-based), (4) calibration methods comparison (Platt / isotonic / temperature), (5) precision/recall/F1 on 4 annotated clinical notes |

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
- After SciSpaCy extraction, a supplemental pass adds colloquial terms BC5CDR commonly misses (`"cold"`, `"flu"`, `"common cold"`, `"nasal allergies"`, `"allergies"`) without overwriting any span SciSpaCy already returned
- Install: see Installation section above

**Fallback — vocabulary matching:**
- ~85 common clinical conditions compiled into a single alternation regex (includes colloquial terms like `"cold"`, `"flu"`, `"common cold"`)
- Ordered longest-to-shortest so `"congestive heart failure"` matches before `"heart failure"` before `"failure"`
- Works with zero setup — used automatically when SciSpaCy is not installed

```python
from src.ner import extract_entities, active_ner_method
print(active_ner_method())          # "scispacy" or "vocabulary"
entities = extract_entities("Patient has hypertension and no evidence of diabetes.")
# → [MedicalEntity("hypertension", 12, 24, "CONDITION"),
#    MedicalEntity("diabetes", 50, 58, "CONDITION")]
```

### `src/coref.py` — Pronoun coreference

When a sentence in a section has no NER entity but contains a pronoun (`it`, `this`, `they`, `the condition`…) and a confident status signal, the status is attributed to the most recently classified entity in the same section.

```text
"The patient had a cough. It resolved."
 ─────────────────────── ────────────
  entity: cough            pronoun + resolved signal
  context: weak (0.35)     → coref fires → cough: resolved
```

**Coref guard rules** (to avoid bad overrides):
- Pronoun sentence confidence must be ≥ 0.70
- Existing entity confidence must be < 0.65 (entity's own sentence was uninformative)
- A 0.92 confidence discount is applied to the pronoun sentence result (acknowledging lost context)
- Coref is section-scoped — a pronoun in Assessment cannot update a PMH entity

### `src/dep_parser.py` — Dependency-tree parsing

Uses spaCy's `en_core_web_sm` parse tree to answer three questions that regex cannot:

**1. Is this entity inside the negation's governing subtree?**

```text
"Patient has no fever but has hypertension."
```

The `no` token governs `fever` (its head's subtree). It does NOT govern `hypertension`. Without dep parsing, a 120-character window around "hypertension" would see "no" and misclassify. With dep parsing, `check_negation_scope(sentence, "hypertension")` returns `False` → the pipeline re-classifies without the negation cue → `ongoing`.

**2. Which conditions were explicitly denied in a list?**

```text
"Patient denies fever, chills, or chest pain."
```

`extract_list_negated(sentence)` finds the denial verb (`denies`), then walks its direct object and all conjuncts to return `["fever", "chills", "chest pain"]` — each marked `negated` individually.

**3. Does a temporal modifier attach to this entity or a subordinate clause?**

```text
"Atrial fibrillation, previously in sinus rhythm."
```

`temporal_modifies_entity(sentence, "atrial fibrillation")` returns `False` — `previously` is inside the `in sinus rhythm` subordinate clause, not attached to the AF noun chunk. The pipeline discards the temporal modifier for AF and re-classifies without it.

All three functions return `None` (not `False`) when the parser is unavailable or the entity is not found in the parse, so they degrade gracefully without spaCy.

### `src/calibration.py` — Platt scaling and calibration analysis

Two capabilities:

**`calibrate(raw_confidence)`** — maps a raw cue-score to an estimated probability of being correct, using a Platt scaler fitted on 2,850 synthetic clinical phrases (88.4% accuracy):

```python
from src.calibration import calibrate
calibrate(0.35)   # → 0.71  (raw 35% → actually correct 71% of the time)
calibrate(0.70)   # → 0.92
calibrate(1.00)   # → 0.97
```

Parameters (`a = 4.1885`, `b = −0.5508`) are stored in `data/calibration.json` and loaded once at first call. Every `classify_condition_status()` call now emits `calibrated_confidence` alongside the raw score. The Streamlit Single Phrase tab displays both.

To re-fit on new data:

```bash
python data/generate_calibration_dataset.py   # regenerates calibration_phrases.csv
# then fit a new logistic regression on (confidence, is_correct) and update calibration.json
```

**`reliability_diagram(csv_path)`** — computes calibration quality over any labelled CSV:

```python
from src.calibration import reliability_diagram
df = reliability_diagram("data/clinical_phrases.csv")
print(f"ECE = {df.attrs['ece']}")   # Expected Calibration Error
# df has columns: bin_center, count, accuracy, avg_confidence, gap
```

**Expected Calibration Error (ECE):** weighted average of |confidence − accuracy| per bin. ECE = 0 is perfect; ECE > 0.10 indicates notable miscalibration. The Streamlit Evaluate tab renders this as a bar chart with an ECE reading.

### `src/note_evaluator.py` — Note-level evaluation

Runs the full pipeline on `data/annotated_notes.json` (4 notes covering mixed PMH/HPI, negation-heavy ROS, pronoun coreference, and temporal signals) and computes precision/recall/F1.

```python
from src.note_evaluator import evaluate_notes
result = evaluate_notes("data/annotated_notes.json")
print(result["aggregate"])
# → {'precision': 0.76, 'recall': 0.76, 'f1': 0.76, 'n_notes': 4, ...}
```

Matching strategy: a pipeline result is a **true positive** if its condition text contains the expected keyword (case-insensitive) AND its status matches. False positives are pipeline detections not matching any annotation; false negatives are expected items not found by the pipeline.

---

## Dataset

`data/clinical_phrases.csv` contains 127 labelled phrases, expanded from the original 15 to include harder cases that expose common failure modes and the real-phrase calibration evaluation set:

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
    "status":               "ongoing",   # ongoing | resolved | negated | ambiguous
    "confidence":           0.82,        # raw cue-score, 0.0 – 1.0
    "calibrated_confidence": 0.95,       # P(correct) from Platt scaler
    "cue":                  "worsening", # highest-weight matched phrase
    "reason":               "Ongoing/active cue found: 'worsening' | Temporal hint: present ('currently')",
    "signals": {
        "negated":          0.0,
        "ambiguous":        0.0,
        "resolved":         0.95,
        "ongoing":          1.0,
        "temporal":         "present",
        "pseudo_negations": [],
        "abbreviations":    ["htn → hypertension"],
        "clause_used":      "full"       # or "final_clause"
    }
}
```

---

## Design Decisions

### Why rule-based rather than ML?

Clinical NLP demands interpretability. Every prediction this system makes can be traced back to a specific cue, a temporal expression, or a section prior. A clinician or engineer auditing a prediction can see exactly why the label was assigned and correct the rules if they disagree.

To make this concrete, a TF-IDF + logistic regression baseline was trained on the same 2,850 synthetic phrases used for Platt calibration, then evaluated on the original 39-phrase hard test set alongside the rule-based system:

| System | Accuracy | Misclassified |
|---|---|---|
| Rule-based (this system) | **97%** | 1 / 39 |
| TF-IDF + logistic regression | 90% | 4 / 39 |

The three cases where the ML baseline fails and the rule system succeeds illustrate exactly why domain engineering matters here:

| Phrase | Gold | ML predicts | Why ML fails |
|---|---|---|---|
| `"Asthma better today"` | `ongoing` | `resolved` | "better" correlates with resolved in training data; rule system knows "better" = improving but still present |
| `"s/p appendectomy"` | `resolved` | `ongoing` | abbreviation not expanded before ML sees it; rule system normalises `s/p → status post` first |
| `"Cough, no improvement noted"` | `ongoing` | `negated` | ML sees "no" and fires negation; rule system masks "no improvement" as a pseudo-negation before scoring |

The ML baseline comparison is available in the **Evaluate Dataset** tab of the Streamlit app.

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

## Calibration Transfer Experiment

### Motivation

Rule-based NLP systems assign confidence scores based on how strongly their rules fire, not on empirical estimates of accuracy. These scores are systematically miscalibrated — a raw confidence of 35% does not mean the system is correct 35% of the time. Correcting this requires a calibration model fitted on labelled data.

The challenge in clinical NLP is that labelled data is scarce and expensive to obtain. This experiment tests the following claim:

> *Miscalibration in rule-based clinical NLP systems is driven by rule activation patterns, not by surface form variation. Calibration models fitted on synthetic template-generated phrases therefore transfer to real clinical text.*

### Experimental setup

| | Detail |
|---|---|
| **Training set** | 2,850 synthetic phrases from `data/calibration_phrases.csv` (template-generated, 88.4% classifier accuracy) |
| **Test set** | 127 manually annotated real phrases from `data/clinical_phrases.csv` (89.8% classifier accuracy) |
| **Methods** | Uncalibrated · Platt scaling · Isotonic regression · Temperature scaling |
| **Metrics** | ECE (Expected Calibration Error) · Brier score |
| **Reproduce** | `python experiments/calibration_transfer.py` |

The three calibration methods are fitted **only on synthetic data** and evaluated **only on real data**. No real phrases are seen during fitting.

### Results

**Overall calibration (127 real phrases):**

| Method | ECE | Brier score | ECE vs raw |
|---|---|---|---|
| Uncalibrated | 0.109 | 0.091 | — |
| Platt scaling | 0.065 | 0.071 | −40% |
| **Isotonic regression** | **0.003** | **0.061** | **−97%** |
| Temperature scaling | 0.130 | 0.095 | +20% |

**Per-category ECE breakdown:**

| Category | n | Uncalibrated | Platt | Isotonic | Temperature |
|---|---|---|---|---|---|
| ongoing | 42 | 0.259 | **0.093** | 0.153 | 0.236 |
| resolved | 38 | 0.116 | 0.083 | **0.032** | 0.102 |
| negated | 22 | 0.142 | 0.115 | **0.095** | 0.157 |
| ambiguous | 25 | 0.140 | 0.176 | **0.116** | 0.164 |

### Interpretation

**1. Isotonic regression transfers best overall (ECE 0.003, −97%).**
A non-parametric monotone function fitted on synthetic confidence–accuracy pairs matches the real data calibration curve almost exactly. This confirms the core claim: what the calibration model needs to learn is the shape of the rule activation–accuracy relationship, which is the same whether the surface text is synthetic or real.

**2. Temperature scaling makes miscalibration worse (ECE 0.130, +20%).**
Temperature scaling applies a single global multiplier (T = 1.5) to all logits. The fact that it fails shows miscalibration is **non-uniform** — the system is overconfident in some confidence ranges and underconfident in others. A single scalar cannot correct this; a bin-specific non-parametric method is required.

**3. Category-level variation reveals a nuanced picture.**
Isotonic regression is not uniformly best across categories. For `ongoing` phrases — which have the widest spread of confidence scores and the most competing signals — Platt scaling (ECE 0.093) outperforms isotonic (ECE 0.153). For `resolved` and `negated`, where high-weight cues produce more consistent scores, isotonic dominates. This suggests a hybrid approach (category-conditional calibration) could reduce ECE further.

**4. Synthetic accuracy (88.4%) ≈ real accuracy (89.8%).**
The near-identical accuracy rates on synthetic and real phrases are consistent with the transfer hypothesis: the rule system's error profile does not change substantially across surface forms, only across genuinely novel phrasing patterns not represented in either dataset.

---

## Limitations and Future Directions

| Limitation | Potential improvement |
|---|---|
| NER misses rare conditions, specialist terminology, and misspellings; colloquial terms (`"cold"`, `"flu"`) are covered by a supplemental pass but uncommon diagnoses are not | Fine-tuned clinical NER model or BERT-based token classifier trained on de-identified notes; broader supplemental vocabulary for common colloquialisms |
| Rule-based classification plateaus on novel phrasing, implicit context, and conditions outside the cue vocabulary | Fine-tuned BERT / clinical LLM — a model trained on labelled clinical notes would generalise far better than hand-crafted rules |
| Platt scaler fitted on 2,850 synthetic phrases — calibration less reliable at the tails | Collect real-world labelled clinical phrases and refit on ≥500 naturally occurring examples |
| Dep-tree heuristics can still mis-scope modifiers in complex multi-clause sentences | Dedicated relation extraction model to link negation and temporality directly to entity spans |
| Pronoun coreference is heuristic — "it" attribution fails when multiple entities are plausible antecedents | Neural coreference resolution (e.g. SpanBERT-based) to handle genuinely ambiguous pronoun references |
| Phrase-level classifier treats multi-entity input as one unit | Sentence-level entity isolation before classification — classify each entity in its own extracted clause |
| Section prior threshold (0.55) is not empirically calibrated | Tune threshold on a held-out annotated note set |
| Low-confidence predictions have no escalation path | Active learning loop — flag predictions below 60% calibrated confidence for clinician review and feed confirmed labels back into the calibration dataset |
| Dep parser silently degrades to regex-only if `en_core_web_sm` is not installed | Surface a clear warning in the UI and CLI when dep parsing is unavailable |

---

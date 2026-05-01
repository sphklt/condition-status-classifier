# Clinical Condition Status Classifier

A clinical NLP system that classifies whether a medical condition in clinical text is **ongoing**, **resolved**, **negated**, or **ambiguous** — at both the phrase level and the full note level.

**Key results** (127-phrase annotated evaluation set):
- Rule-based classifier: **89.8% accuracy**
- Hybrid triage: **100% recall of errors**, 62% of predictions auto-approved at 100% accuracy
- Calibration transfer: isotonic regression reduces ECE from 0.109 → 0.003 (**−97%**) using only synthetic training data
- TAM: grammatical tense/aspect/modality adds **+0.8% accuracy**, ECE −4%, zero predictions hurt
- 254 tests across 7 test files

---

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest                      # 254 passed
streamlit run app.py        # browser UI
python main.py              # CLI evaluation
```

**Optional — SciSpaCy NER** (improves entity recall on real notes; requires Python 3.11):

```bash
pip install scispacy==0.5.4
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

Without SciSpaCy, the system uses a vocabulary-based NER covering ~85 common clinical conditions with no setup required.

---

## Status Labels

| Label | Meaning | Examples |
|---|---|---|
| `ongoing` | Condition is active, persistent, stable, controlled, or currently changing | `"Persistent cough for 2 weeks"`, `"Diabetes is stable"`, `"Seizures controlled on medication"` |
| `resolved` | Condition is historical, closed, or no longer active | `"History of asthma"`, `"Fever has resolved"`, `"s/p appendectomy"` |
| `negated` | Condition is explicitly denied or absent | `"Patient denies chest pain"`, `"No evidence of pneumonia"`, `"Fever -ve"` |
| `ambiguous` | Condition is uncertain, suspected, or unconfirmed | `"Possible pneumonia"`, `"Rule out sepsis"`, `"Concern for pulmonary embolism"` |

**Important nuance — "better" is ongoing, not resolved:**

```
Asthma better today  →  ongoing
```

"Better" means the condition has improved but is still present. Words like `stable`, `controlled`, `improving`, and `better` all signal an active condition being managed — not a resolved one.

---

## How It Works

### Five non-trivial capabilities

**1. Pseudo-negation filtering (NegEx-inspired)**

Some phrases contain negation words that do not deny a condition. These patterns are detected and masked before cue scoring.

| Phrase | Naive | This system | Why |
|---|---|---|---|
| `No longer has headache` | `negated` | `resolved` | "no longer" = condition ended, not denied |
| `Not improving on current regimen` | `negated` | `ongoing` | condition present, not responding |
| `No improvement noted` | `negated` | `ongoing` | condition persists unchanged |
| `No change in diabetes status` | `negated` | `ongoing` | unchanged = still present |

**2. Adversative clause detection**

When a sentence has clauses separated by an adversative conjunction or a period, the final clause carries the definitive status.

```
"I had severe flu which I think is getting better now.
 But after a couple of days, it got completely over."
→ resolved  (final clause wins; first clause ignored)
```

**3. Temporal signal detection**

Past and present temporal expressions boost the relevant label score independently of keyword cues.

| Phrase | Signal | Effect |
|---|---|---|
| `"DM diagnosed 3 years ago"` | past | boosts resolved |
| `"Chest pain since this morning"` | present | boosts ongoing |
| `"Was treated for pneumonia last year"` | past | boosts resolved |
| `"Acute onset shortness of breath"` | present | boosts ongoing |

**4. Sentence-boundary-aware context windows**

Entity classification uses only the sentence containing the entity — not a fixed character window. This prevents signals from adjacent sentences from bleeding into the wrong classification.

```
"No fever.  Patient has diabetes."
 ─────────  ─────────────────────
 sentence 1   sentence 2 → "No" from sentence 1 never reaches diabetes
```

**5. Pronoun coreference within sections**

When a pronoun sentence contains a confident status signal and the entity's own sentence was uninformative, the status is attributed to the most recent entity in the same section.

```
"The patient had a cough. It resolved."
                          ──────────── → cough: resolved (87% conf)
```

Coref fires only when the entity's own sentence confidence < 0.65, ensuring it never overrides a confident classification. Coref is section-scoped.

### Design decisions

**Why rule-based rather than ML?**

Every prediction can be traced to a specific cue, temporal expression, or section prior. An ML baseline (TF-IDF + logistic regression, trained on the same 2,850 synthetic phrases) shows exactly where domain engineering wins:

| System | Accuracy | Misclassified |
|---|---|---|
| Rule-based (this system) | **97%** | 1 / 39 |
| TF-IDF + logistic regression | 90% | 4 / 39 |

| Phrase | Gold | ML predicts | Why ML fails |
|---|---|---|---|
| `"Asthma better today"` | `ongoing` | `resolved` | "better" correlates with resolved in training data |
| `"s/p appendectomy"` | `resolved` | `ongoing` | abbreviation not expanded before ML sees it |
| `"Cough, no improvement noted"` | `ongoing` | `negated` | ML sees "no"; rule system masks pseudo-negation |

**Why weighted multi-signal scoring rather than priority order?**

The original system used fixed priority: negated → ambiguous → resolved → ongoing. This fails when two signals appear in the same phrase:

```
"History of asthma, currently worsening"
 → Priority order: resolved (first cue found)
 → Weighted scoring: ongoing (worsening 0.95 + currently 0.90 + bonus > history of 0.95)
```

**Why a two-level pipeline?**

The phrase classifier is optimised for short, focused input. The pipeline feeds it exactly what it needs — a single sentence from the appropriate section — without the classifier needing to know how sections, NER, or sentence splitting work.

---

## Research

### Tense-Aspect-Modality (TAM) Classification

**Motivation.** All existing clinical condition classifiers — including this system's rule-based and Bayesian layers — rely on *lexical* signals: keyword lookup with associated weights. Grammatical predicate structure carries strong status information that no keyword can capture:

- `"Symptoms are worsening"` — no temporal adverb, but present progressive = ongoing
- `"Blood pressure should be monitored"` — no cue fires, but deontic modal = ongoing obligation
- `"The infection had resolved"` — past perfect = completed before a past reference point → resolved
- `"Symptoms might be worsening"` — strong ongoing cue, but epistemic modal raises uncertainty

**Approach.** Extract the Tense-Aspect-Modality signature of the predicate governing the condition, then map each component to an independent log-likelihood ratio (LLR) that feeds the same Bayesian log-score vector as keyword cues:

```
TAMSignature(tense, aspect, modal)
  tense:  past | present | future
  aspect: progressive | perfect | past_perfect
  modal:  epistemic_weak | epistemic_strong | deontic | negated_deontic | conditional
```

**Key novelty — compositionality:** tense, aspect, and modality each contribute independent LLRs. Novel constructions are handled correctly without explicit cue entries:

```
"might have been resolving"
  epistemic_weak (modal)  → shifts posterior toward ambiguous
  + progressive (aspect)  → shifts toward ongoing
  = high-entropy posterior (ambiguous vs ongoing) → triage flagged
```

This is categorically different from adding "might have been resolving" as a cue phrase — it generalises to any unseen combination.

**Complementarity with temporal detection.** `temporal.py` catches adverbial time expressions ("3 years ago", "currently"). TAM catches grammatical predicate structure. They are independent evidence sources:

| Signal type | Example | What fires |
|---|---|---|
| Adverbial temporal | `"diagnosed 3 years ago"` | temporal.py |
| Grammatical TAM | `"is worsening"` | TAM (progressive) |
| Both | `"was stable last month"` | temporal + TAM (past) |
| Neither | `"no evidence of fever"` | keyword cue |

**Safety constraint.** Tense patterns are intentionally conservative — only specific clinical verb constructions, never bare `is/has/was` — to prevent false positives in negation contexts. `"Patient had no fever"` fires no TAM signal and stays `negated`; `"The fever had resolved"` fires `past_perfect` and boosts `resolved`.

**Results** (127-phrase set, with vs without TAM):

| Metric | Without TAM | With TAM | Δ |
|---|---|---|---|
| Accuracy | 87.4% | **88.2%** | +0.8% |
| ECE | 0.125 | **0.120** | −4% |
| Mean entropy | 0.886 bits | **0.849 bits** | −4.2% |
| Predictions changed | — | 1 improved, 0 hurt | — |

TAM fires on 16/127 (13%) phrases. The one prediction it changes: `"Symptoms may represent early heart failure"` — `ongoing` without TAM → `ambiguous` with TAM (correct; `may` = epistemic_weak).

Reproduce: `python experiments/tam_eval.py`

---

### Calibration Transfer

**Motivation.** Rule-based systems assign confidence scores based on how strongly rules fire, not empirical accuracy. These scores are systematically miscalibrated. The key question: can a calibration model fitted on *synthetic* template-generated phrases transfer to *real* clinical text?

> *Miscalibration is driven by rule activation patterns, not surface form variation. Calibration models therefore transfer from synthetic to real text.*

**Setup:**

| | |
|---|---|
| Training set | 2,850 synthetic phrases (`data/calibration_phrases.csv`); 88.4% classifier accuracy |
| Test set | 127 real phrases (`data/clinical_phrases.csv`); 89.8% classifier accuracy |
| Methods | Uncalibrated · Platt scaling · Isotonic regression · Temperature scaling |

**Results:**

| Method | ECE | Brier | ECE vs raw |
|---|---|---|---|
| Uncalibrated | 0.109 | 0.091 | — |
| Platt scaling | 0.065 | 0.071 | −40% |
| **Isotonic regression** | **0.003** | **0.061** | **−97%** |
| Temperature scaling | 0.130 | 0.095 | +20% |

Per-category ECE:

| Category | n | Uncalibrated | Platt | Isotonic | Temperature |
|---|---|---|---|---|---|
| ongoing | 42 | 0.259 | **0.093** | 0.153 | 0.236 |
| resolved | 38 | 0.116 | 0.083 | **0.032** | 0.102 |
| negated | 22 | 0.142 | 0.115 | **0.095** | 0.157 |
| ambiguous | 25 | 0.140 | 0.176 | **0.116** | 0.164 |

**Interpretation.** Isotonic regression transfers nearly perfectly (ECE 0.003), confirming the transfer hypothesis. Temperature scaling fails (ECE +20%) because miscalibration is non-uniform — a single global scalar cannot correct category-specific over/under-confidence. For `ongoing`, Platt scaling (ECE 0.093) outperforms isotonic (0.153) due to wider score spread; isotonic dominates for `resolved` and `negated`.

Reproduce: `python experiments/calibration_transfer.py`

---

### Bayesian Evidence Fusion

**Motivation.** The rule-based classifier returns a single confidence score and argmax — no distribution, no uncertainty. Bayesian fusion treats cue weights as calibrated likelihood ratios and accumulates evidence into a proper posterior.

**Algorithm.** For each label ℓ ∈ {ongoing, resolved, negated, ambiguous}:

```
log_score[ℓ] = log P(label=ℓ | section)       ← section-conditional prior
             + Σ log(w / (1-w))                ← for each cue targeting ℓ
             - Σ log(w' / (1-w')) / 3          ← for each cue targeting ℓ' ≠ ℓ

posterior[ℓ] = softmax(log_score)[ℓ]
```

Cue weights (w) are already calibrated precision estimates from `rules.py`, so calibration propagates at the feature level. Section priors encode clinical domain knowledge (PMH → resolved=0.55; HPI/Assessment → ongoing=0.50). Temporal signals contribute their own LLRs (e.g., past temporal → +0.85 to resolved). Cues with w ≤ 0.50 are skipped (LLR ≤ 0 = no positive evidence).

**Results** (127 real phrases):

| System | Accuracy | ECE |
|---|---|---|
| Rule-based | **89.8%** | 0.109 |
| Bayesian fusion | 87.4% | 0.125 |

Per-label ECE — Bayesian wins on 3 of 4 categories:

| Label | Rule ECE | Bayes ECE |
|---|---|---|
| ongoing | 0.259 | **0.215** |
| resolved | 0.116 | **0.104** |
| negated | 0.142 | **0.131** |
| ambiguous | **0.140** | 0.190 |

**Entropy as triage signal:**

| Prediction type | Mean entropy |
|---|---|
| Correct predictions | 0.75 bits |
| Wrong predictions | 1.85 bits |
| Ratio | **2.5× higher on errors** |

Wrong predictions have 2.5× higher entropy than correct ones, making entropy a reliable flag for routing uncertain predictions to human review.

Reproduce: `python experiments/bayesian_fusion_eval.py`

---

### Hybrid Classifier

`src/hybrid.py` combines the rule-based MAP label with Bayesian uncertainty into a single triage-aware result.

**Triage logic:** a prediction is flagged when either:
- Bayesian entropy > 1.2 bits (tunable default)
- The two systems predict different labels

```python
from src.hybrid import classify

result = classify("History of poorly controlled hypertension.")
# result["status"]       → "resolved"       (rule-based MAP)
# result["posterior"]    → {"ongoing": 0.47, "resolved": 0.38, ...}
# result["entropy"]      → 1.21 bits        (two signals compete)
# result["triage_flag"]  → True             (flagged for review)
# result["runner_up"]    → ("ongoing", 0.47)
```

**Triage performance** (127-phrase set, default threshold 1.2 bits):

| Metric | Value |
|---|---|
| Phrases flagged | 48 / 127 (38%) |
| Recall of errors | **100%** — every wrong prediction flagged |
| Precision | 27% — 1 in 3.7 flagged phrases is an actual error |
| Auto-approved accuracy | **100%** — 62% of predictions, no errors |
| Review efficiency | **2.6×** fewer phrases read per error |

At best-F1 threshold (1.8 bits): 21% flagged, 77% recall, 37% precision.

The threshold is tunable. `evaluate_triage(csv_path, thresholds=[...])` sweeps thresholds and returns precision/recall/F1 at each.

Reproduce: `python experiments/hybrid_eval.py`

---

## Architecture

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
Platt calibration           raw cue-score → P(correct)
    │
    ▼
{status, confidence, calibrated_confidence, cue, reason, signals}
```

### Level 1b — Hybrid Classifier

```
Raw phrase + section
    │
    ├──► Rule-based classifier ──► status, confidence, calibrated_confidence
    │
    └──► Bayesian fusion ──────► posterior {label: P}, entropy (bits)
                │
                ▼
         Triage decision    entropy > 1.2 bits OR systems disagree → triage_flag=True
                │
                ▼
{status, posterior, entropy, runner_up, triage_flag, triage_reason, agreement}
```

### Level 2 — Full Note Pipeline

```
Raw clinical note
    │
    ▼
Section detector            PMH → resolved prior | HPI → ongoing prior
    │
    ▼  (per section)
Abbreviation normalizer  →  Sentence splitter  →  NER
    │
    ▼
Context extraction          sentence containing each entity (not fixed char window)
    │
    ▼
Phrase classifier  →  Dep parser refinement  →  Section prior override
    │
    ▼
Pronoun coreference  →  Deduplication
    │
    ▼
[{condition, status, confidence, section, reason}, ...]
```

---

## Usage

### Single phrase

```python
from src.classifier import classify_condition_status
from src.hybrid import classify

# Rule-based only
classify_condition_status("History of asthma, currently worsening")
# → {"status": "ongoing", "confidence": 1.0, "calibrated_confidence": 0.97, ...}

# Hybrid (rule-based + Bayesian uncertainty)
classify("No evidence of pneumonia.")
# → {"status": "negated", "triage_flag": False, "entropy": 0.0, ...}
```

### Full clinical note

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
"""
print(format_results(process_note(note)))
```

```
CONDITION                           STATUS         CONF  SECTION
----------------------------------------------------------------
dyspnea                             ongoing        100%  history_of_present_illness
chest pain                          negated        100%  history_of_present_illness
fever                               negated        100%  history_of_present_illness
Hypertension                        resolved       100%  past_medical_history
type 2 diabetes mellitus            resolved       100%  past_medical_history
pneumonia                           resolved       100%  past_medical_history
atrial fibrillation                 resolved       100%  past_medical_history
heart failure                       ambiguous       76%  assessment
pulmonary embolism                  ambiguous       95%  assessment
```

### Output schema

**`classify_condition_status(text)`:**

```python
{
    "status":                "ongoing",
    "confidence":            0.82,
    "calibrated_confidence": 0.95,
    "cue":                   "worsening",
    "reason":                "Ongoing/active cue found: 'worsening' | Temporal hint: present",
    "signals": {
        "negated": 0.0, "ambiguous": 0.0, "resolved": 0.95, "ongoing": 1.0,
        "temporal": "present", "pseudo_negations": [], "clause_used": "full"
    }
}
```

**`hybrid.classify(text, section)`:**

```python
{
    "status":                "ongoing",
    "confidence":            0.82,
    "calibrated_confidence": 0.95,
    "posterior":             {"ongoing": 0.71, "resolved": 0.18, "negated": 0.07, "ambiguous": 0.04},
    "entropy":               0.43,
    "runner_up":             ("resolved", 0.18),
    "agreement":             True,
    "triage_flag":           False,
    "triage_reason":         "",
    "rule_reason":           "Ongoing/active cue found: 'worsening'",
    "rule_cue":              "worsening",
    "bayes_status":          "ongoing",
    "signals":               { ... }
}
```

### Streamlit app

```bash
streamlit run app.py
```

| Tab | What it does |
|---|---|
| Single Phrase | Classify a phrase; shows triage flag, posterior bar chart, runner-up, confidence, expanded abbreviations |
| Full Clinical Note | Paste a note; runs the full pipeline and returns a colour-coded condition table |
| Evaluate Dataset | Phrase accuracy, reliability diagram + ECE, ML baseline comparison, calibration methods comparison, note-level P/R/F1 |

---

## Project Structure

```
condition-status-classifier/
│
├── data/
│   ├── clinical_phrases.csv          127-phrase labelled dataset
│   ├── annotated_notes.json          4 annotated clinical notes for P/R/F1 evaluation
│   ├── calibration.json              fitted Platt scaler parameters
│   ├── calibration_phrases.csv       2,850 synthetic phrases for calibration fitting
│   └── generate_calibration_dataset.py
│
├── src/
│   ├── normalizer.py                 abbreviation expansion
│   ├── rules.py                      weighted cues (100+ entries) + pseudo-negation patterns
│   ├── temporal.py                   past/present temporal signal detection
│   ├── classifier.py                 phrase-level classifier
│   ├── dep_parser.py                 spaCy dep-tree: negation scope, list negation, temporal scope
│   ├── section_detector.py           note section splitter
│   ├── ner.py                        NER (SciSpaCy primary / vocabulary fallback)
│   ├── sentence_splitter.py          clinical sentence boundary detection
│   ├── coref.py                      pronoun-to-entity coreference within sections
│   ├── pipeline.py                   full note pipeline
│   ├── calibration.py                Platt scaler + ECE + calibration transfer helpers
│   ├── tam.py                        TAM extraction (tense/aspect/modality → LLRs)
│   ├── bayesian_fusion.py            Bayesian evidence fusion (posterior + entropy + TAM)
│   ├── hybrid.py                     hybrid classifier (rule-based MAP + Bayesian triage)
│   ├── baseline.py                   TF-IDF + logistic regression baseline
│   ├── note_evaluator.py             pipeline P/R/F1 on annotated notes
│   └── utils.py                      phrase-level dataset evaluation
│
├── experiments/
│   ├── calibration_transfer.py
│   ├── bayesian_fusion_eval.py
│   ├── hybrid_eval.py
│   └── tam_eval.py
│
├── tests/
│   ├── test_classifier.py            33 tests
│   ├── test_pipeline.py              44 tests
│   ├── test_coref.py                 21 tests
│   ├── test_dep_and_calibration.py   38 tests
│   ├── test_bayesian_fusion.py       36 tests
│   ├── test_hybrid.py                31 tests
│   └── test_tam.py                   52 tests
│
├── app.py
├── main.py
├── pytest.ini
└── requirements.txt
```

### Module reference

| Module | Role |
|---|---|
| `src/normalizer.py` | Expands clinical abbreviations (`h/o → history of`, `s/p → status post`, `-ve → negative for`) |
| `src/rules.py` | Weighted cue lists (100+ entries) ordered most-to-least specific + pseudo-negation registry |
| `src/temporal.py` | Past/present temporal regex patterns; returns signal and confidence |
| `src/classifier.py` | Orchestrates phrase-level classification; emits `calibrated_confidence` |
| `src/dep_parser.py` | spaCy dependency parsing: negation scope, list negation, temporal modifier scope |
| `src/section_detector.py` | Splits clinical notes into labeled sections; drives section-conditional priors |
| `src/ner.py` | SciSpaCy primary NER + vocabulary fallback (~85 conditions) |
| `src/sentence_splitter.py` | Clinical sentence boundary detection with abbreviation protection |
| `src/coref.py` | Pronoun coreference within sections; fires only when entity confidence < 0.65 |
| `src/pipeline.py` | Orchestrates the full note-level pipeline; wires dep parser refinement |
| `src/calibration.py` | Platt scaler (`calibrate()`), reliability diagram, ECE, Brier, isotonic/temperature methods |
| `src/tam.py` | TAM extraction: tense/aspect/modality → LLRs; compositionality over novel predicate constructions |
| `src/bayesian_fusion.py` | Bayesian evidence fusion: posterior distribution, Shannon entropy, section priors, TAM integration |
| `src/hybrid.py` | Hybrid classifier: rule-based MAP label + Bayesian posterior + entropy triage flag |
| `src/baseline.py` | TF-IDF + logistic regression baseline |
| `src/note_evaluator.py` | Precision/recall/F1 evaluation on annotated clinical notes |
| `src/utils.py` | Phrase-level dataset evaluation helper |

---

## Limitations

| Limitation | Potential improvement |
|---|---|
| NER misses rare conditions, specialist terminology, and misspellings | Fine-tuned clinical NER or BERT-based token classifier trained on de-identified notes |
| Rule-based classification plateaus on novel phrasing and conditions outside the cue vocabulary | Fine-tuned BERT / clinical LLM trained on labelled clinical notes |
| Platt scaler fitted on 2,850 synthetic phrases — less reliable at the tails | Collect ≥500 real labelled phrases and refit |
| Dep-tree heuristics can mis-scope modifiers in complex multi-clause sentences | Dedicated relation extraction model linking negation/temporality to entity spans |
| Pronoun coreference fails when multiple entities are plausible antecedents | Neural coreference resolution (e.g. SpanBERT-based) |
| Section prior threshold (0.55) is not empirically calibrated | Tune on a held-out annotated note set |
| Dep parser silently degrades to regex-only if `en_core_web_sm` is not installed | Surface a clear warning in the UI and CLI |

---

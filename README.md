# Clinical Condition Status Classifier

A clinical NLP system that classifies whether a medical condition in clinical text is **ongoing**, **resolved**, **negated**, or **ambiguous** — at both the phrase level and the full note level.

**Key results** (159-phrase annotated evaluation set):
- Rule-based classifier: **89.8% accuracy**
- Hybrid triage: **100% recall of errors**, 62% of predictions auto-approved at 100% accuracy
- Calibration transfer: isotonic regression reduces ECE from 0.109 → 0.003 (**−97%**) using only synthetic training data
- TAM: grammatical tense/aspect/modality adds **+13.9% accuracy** (ambiguous label +40%), 21 improved, 0 hurt
- Trajectory: intra-section status tracking adds **+40% accuracy** on multi-mention conditions; note-level F1 **0.694 → 0.757** (+9%)
- Attribution: asserter-identity signals add **+4.4% accuracy** on 159-phrase set; inline attribution cases 40% → **96%** (+56%)
- 332 tests across 9 test files

---

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest                      # 332 passed
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

**Results** (151-phrase set, with vs without TAM):

| Metric | Without TAM | With TAM | Δ |
|---|---|---|---|
| Accuracy | 76.2% | **90.1%** | **+13.9%** |
| ECE | 0.065 | 0.157 | +0.091 |
| Mean entropy | 1.012 bits | **0.915 bits** | −9.6% |
| Predictions changed | — | **21 improved, 0 hurt** | — |

Per-label accuracy on TAM-sensitive constructions (64 new phrases):

| Label | Without TAM | With TAM | Δ |
|---|---|---|---|
| ambiguous | 48% | **88%** | **+40%** |
| resolved | 84% | **93%** | +9% |
| ongoing | 90% | 90% | 0% |
| negated | 86% | 86% | 0% |

TAM fires on 40/151 (26%) phrases. The 21 improved predictions split across epistemic modals correctly routed to `ambiguous` and past-perfect constructions correctly resolved without any explicit resolved keyword. Zero predictions hurt.

ECE increases because the system becomes much more confident on the new phrases — and while those confidences are directionally correct (21 improvements), the isotonic calibrator was fitted on the original 127-phrase distribution and does not re-calibrate for the expanded set.

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

### Status Trajectory

**Motivation.** Every prior approach — this system included — classifies each entity against the *single sentence* that contains it. When the same condition appears multiple times within a section, earlier sentences are discarded. This throws away directional evidence: a condition mentioned as ongoing in sentence 1 and resolved in sentence 3 is a *resolution*, not an ambiguous signal.

**Novel contribution.** Model the sequence of status classifications for a condition across all sentences in a section, then reconcile the sequence using time-decayed log-evidence accumulation. Each transition type (resolution, relapse, contradiction, clarification) earns a bonus LLR encoding the clinical prior that this transition is meaningful:

| Transition | Clinical meaning | Bonus LLR |
|---|---|---|
| `ongoing → resolved` | resolution — condition cleared | +0.80 |
| `resolved → ongoing` | relapse — history item now active | +1.00 |
| `negated → ongoing` | contradiction — prior denial overridden | +0.90 |
| `ambiguous → [definite]` | clarification — uncertainty resolved | +0.60 |

**Time-decay reconciliation.** Each point contributes `0.7^(n−1−i)` weight — the most recent mention dominates but earlier signals still inform. This is strictly better than "take the last sentence" because early-context confidence bounds the posterior when the final sentence is weak.

```python
# Resolution: 9/10 test cases resolved correctly vs. 1/10 baseline
"Patient has cough. Cough resolved after antibiotic treatment."
  Point 0: ongoing (0.80)  →  Point 1: resolved (0.88)
  Transition bonus: +0.80 to resolved
  → Final: resolved (trajectory), ongoing (baseline, wrong)
```

**Key property — transition type as triage signal.** The `transition_type` field (`relapse`, `contradiction`, `multi_transition`) feeds directly into the hybrid triage system: a relapse or contradiction detected by trajectory is surfaced for clinical review regardless of entropy.

**Results** (40 multi-sentence passages, all with 2 explicit entity mentions):

| Transition type | n | Baseline | Trajectory | Δ |
|---|---|---|---|---|
| resolution | 10 | 10% | **100%** | +90% |
| relapse | 10 | 10% | **40%** | +30% |
| contradiction | 5 | 0% | **40%** | +40% |
| clarification | 5 | 0% | **40%** | +40% |
| stable | 10 | 100% | **100%** | 0% |
| **Overall** | **40** | **30%** | **70%** | **+40%** |

16 improved, 0 hurt. Trajectory never degrades a correct single-sentence prediction.

**Note-level results** (7-note annotated set, with vs without trajectory):

| Metric | Without trajectory | With trajectory | Δ |
|---|---|---|---|
| Precision | 0.625 | **0.684** | +9.4% |
| Recall | 0.781 | **0.848** | +8.6% |
| F1 | 0.694 | **0.757** | +9.1% |

14/14 trajectory-dependent conditions correctly classified across the 3 new notes (resolution, relapse, clarification types). The 7 FN errors in both configurations are pre-existing NER misses unrelated to status classification.

**Complementarity with coref.** Pronoun coreference (`coref.py`) handles `"Patient has cough. It resolved."` (pronoun bridge). Trajectory handles `"Patient has cough. Cough resolved."` (explicit re-mention). Neither replaces the other — both fire in the same pipeline step.

Reproduce: `python experiments/trajectory_eval.py`

---

### Attribution-Aware Confidence

**Motivation.** TAM captures *how* the predicate is expressed (past perfect, epistemic modal). Temporal signals capture *when* the event occurred. Both are silent about *who* is asserting the status. Clinical text carries a fourth independent evidence dimension: the asserter's identity. A patient who *thinks* they have hypertension, a family member who *reports* a seizure history, and an EHR fragment that *shows* diabetes are three qualitatively different claims — and none of them are captured by keyword, temporal, or TAM signals.

**Key novel case — record attribution.** Phrases like `"Per records, hypertension"` or `"Records show asthma"` contain no resolved keyword, no temporal adverb, and no grammatical past signal. Without attribution, the classifier has no resolved evidence and defaults to ongoing/ambiguous. The record source contributes LLR +1.0 to resolved — enough to produce the correct classification from a single signal.

**Approach.** Extract the asserter's identity from the sentence, classify it into one of five attribution source types, and map each source to an independent LLR vector that feeds the same Bayesian log-score accumulator as keyword, temporal, and TAM signals:

| Source | Clinical meaning | Primary LLR effect |
|---|---|---|
| `record` | per records/chart, records show/document | +1.0 to resolved |
| `patient_hedge` | patient thinks/believes/suspects | +0.80 to ambiguous |
| `clinician_hedge` | we think/believe, appears consistent with | +0.60 to ambiguous |
| `family_report` | family/wife/caregiver reports/states | +0.40 to ambiguous |
| `patient_report` | patient reports/states/endorses | +0.30 to ambiguous |

**Priority ordering.** When multiple patterns overlap, `patient_hedge` takes priority over `patient_report` to prevent a weak report signal from masking a stronger hedge: `"Patient thinks she reports having anxiety"` → `patient_hedge`.

**Safety constraint.** All attribution LLRs are capped at ±1.0. Strong keyword cues (LLR ≈ 6.9 for weight=0.999) always dominate attribution — `"Patient denies fever"` stays `negated` even though `patient_report` fires, because `denies` (LLR ≈ 3.7) overwhelms attribution (+0.30).

**False-positive prevention for family history.** `"Family history of diabetes"` contains "family" but has no report verb — the pattern requires `family/wife/... + reports?/states?/mentions?` to prevent family history entries from incorrectly triggering `family_report`.

**Results** (159-phrase dataset, without vs with attribution):

| Metric | Without attribution | With attribution | Δ |
|---|---|---|---|
| Accuracy | 86.2% | **90.6%** | **+4.4%** |
| Changed predictions | — | **7 improved, 0 hurt** | — |

Attribution fires on only 8/159 (5%) of the dataset phrases — all `record` source — but converts every one from incorrect (ongoing) to correct (resolved). Zero predictions hurt.

Attribution-only inline evaluation (n=25 targeted cases):

| Source | n | Without attribution | With attribution | Δ |
|---|---|---|---|---|
| record | 12 | 25% | **100%** | +75% |
| patient_hedge | 4 | 50% | **100%** | +50% |
| clinician_hedge | 4 | 0% | **75%** | +75% |
| patient_report | 5 | 100% | 100% | 0% |
| **Overall** | **25** | **40%** | **96%** | **+56%** |

`patient_report` baseline is already 100% because the ongoing cue in `"Patient reports chest pain"` independently scores the phrase correctly — the attribution signal is additive but not load-bearing. `clinician_hedge` (75%) misses one case where the hedged phrase still has a strong keyword cue dominating.

**Complementarity with TAM and temporal signals.** All three are orthogonal evidence dimensions feeding the same log-score vector:

| Signal type | Example | What fires |
|---|---|---|
| Keyword cue | `"resolved after treatment"` | `rules.py` |
| Temporal adverb | `"diagnosed 3 years ago"` | `temporal.py` |
| Grammatical TAM | `"had resolved"` (past perfect) | `tam.py` |
| Asserter attribution | `"per records, hypertension"` | `attribution.py` |

Reproduce: `python experiments/attribution_eval.py`

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
Pronoun coreference  →  Trajectory refinement  →  Deduplication
    │
    ▼
[{condition, status, confidence, section, reason, trajectory?}, ...]
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
│   ├── clinical_phrases.csv          159-phrase labelled dataset (incl. TAM-sensitive, record attribution)
│   ├── annotated_notes.json          7 annotated clinical notes (4 original + 3 trajectory-specific)
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
│   ├── trajectory.py                 intra-section status trajectory (time-decay reconciliation)
│   ├── pipeline.py                   full note pipeline (incl. trajectory refinement)
│   ├── calibration.py                Platt scaler + ECE + calibration transfer helpers
│   ├── tam.py                        TAM extraction (tense/aspect/modality → LLRs)
│   ├── attribution.py                attribution-aware confidence (asserter identity → LLRs)
│   ├── bayesian_fusion.py            Bayesian evidence fusion (posterior + entropy + TAM + attribution)
│   ├── hybrid.py                     hybrid classifier (rule-based MAP + Bayesian triage)
│   ├── baseline.py                   TF-IDF + logistic regression baseline
│   ├── note_evaluator.py             pipeline P/R/F1 on annotated notes
│   └── utils.py                      phrase-level dataset evaluation
│
├── experiments/
│   ├── calibration_transfer.py
│   ├── bayesian_fusion_eval.py
│   ├── hybrid_eval.py
│   ├── tam_eval.py
│   ├── trajectory_eval.py
│   └── attribution_eval.py
│
├── tests/
│   ├── test_classifier.py            33 tests
│   ├── test_pipeline.py              44 tests
│   ├── test_coref.py                 21 tests
│   ├── test_dep_and_calibration.py   38 tests
│   ├── test_bayesian_fusion.py       36 tests
│   ├── test_hybrid.py                31 tests
│   ├── test_tam.py                   52 tests
│   ├── test_trajectory.py            29 tests
│   └── test_attribution.py           49 tests
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
| `src/trajectory.py` | Intra-section status trajectory: time-decayed log-evidence accumulation + transition-type LLR bonuses |
| `src/pipeline.py` | Orchestrates the full note-level pipeline; wires dep parser, coref, and trajectory refinement |
| `src/calibration.py` | Platt scaler (`calibrate()`), reliability diagram, ECE, Brier, isotonic/temperature methods |
| `src/tam.py` | TAM extraction: tense/aspect/modality → LLRs; compositionality over novel predicate constructions |
| `src/attribution.py` | Attribution-aware confidence: extracts asserter identity (record/patient/family/clinician) and maps each source to independent LLRs |
| `src/bayesian_fusion.py` | Bayesian evidence fusion: posterior distribution, Shannon entropy, section priors, TAM + attribution integration |
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

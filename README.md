# Clinical Condition Status Classifier

A small, explainable clinical NLP project that classifies whether a medical condition mentioned in a short clinical phrase is **ongoing**, **resolved**, **negated**, or **ambiguous**.

This project was built as a rule-based baseline for condition status detection in clinical text. The goal is to demonstrate clear clinical reasoning, reproducible code, evaluation, tests, and an optional Streamlit demo.

---

## Problem

Clinical text often mentions conditions in different ways. A condition may be currently active, historical, denied by the patient, or uncertain.

For example:

| Clinical Phrase | Expected Status | Reason |
|---|---|---|
| `The patient has asthma` | `ongoing` | The condition appears currently active. |
| `Asthma better today` | `ongoing` | The condition has improved but is not resolved. |
| `Asthma resolved after treatment` | `resolved` | The phrase contains an explicit resolution cue. |
| `Patient denies chest pain` | `negated` | The patient denies the symptom. |
| `Possible pneumonia` | `ambiguous` | The diagnosis is uncertain. |

The task is to infer condition status from the phrase.

---

## Project Goal

The goal of this project is to build a simple, interpretable baseline system that can:

1. Read short clinical phrases.
2. Detect clinical status cues.
3. Classify each phrase into a condition status label.
4. Provide the matched cue and explanation for the prediction.
5. Evaluate predictions against a small labeled dataset.
6. Provide a lightweight Streamlit interface for interactive testing.

---

## Status Labels

This project uses four labels.

### 1. `ongoing`

The condition appears active, chronic, persistent, controlled, stable, improving, or worsening.

Examples:

```text
The patient has asthma
Asthma better today
Diabetes is stable
Persistent cough for 2 weeks
Seizures controlled on medication
```

Important note:

```text
Asthma better today
```

is classified as `ongoing`, not `resolved`, because “better” means the condition has improved, but it does not indicate that the condition is fully gone.

---

### 2. `resolved`

The condition appears closed, historical, or no longer active.

Examples:

```text
Asthma resolved after treatment
Fever has resolved
History of asthma
Previous fracture of left arm
No longer has headache
```

---

### 3. `negated`

The condition is explicitly denied or absent.

Examples:

```text
Patient denies chest pain
No evidence of pneumonia
Negative for abdominal pain
```

---

### 4. `ambiguous`

The condition is uncertain, suspected, or requires further review.

Examples:

```text
Possible pneumonia
Rule out sepsis
Concern for infection
Suspected asthma
```

---

## Approach

This project uses a rule-based classifier.

The system looks for cue phrases in the input text and assigns a label based on priority.

Priority order:

1. Negation cues
2. Ambiguity cues
3. Resolved or historical cues
4. Ongoing or active cues
5. Default to ongoing if no cue is found

This priority is important because some clinical phrases can contain overlapping signals.

Example:

```text
No evidence of active pneumonia
```

Even though the phrase contains the word `active`, the stronger cue is `no evidence of`, so the condition should be classified as `negated`.

---

## Rule Categories

### Negation cues

Examples:

```text
denies
no evidence of
negative for
without
no signs of
not present
```

### Resolved cues

Examples:

```text
resolved
no longer
history of
past history of
previous
prior
status post
s/p
```

### Ongoing cues

Examples:

```text
has
currently
active
persistent
ongoing
worsening
stable
controlled
improving
better
```

### Ambiguous cues

Examples:

```text
possible
rule out
r/o
suspected
may have
concern for
question of
```

---

## Project Structure

```text
condition-status-classifier/
│
├── data/
│   └── clinical_phrases.csv
│
├── src/
│   ├── __init__.py
│   ├── rules.py
│   ├── classifier.py
│   └── utils.py
│
├── tests/
│   └── test_classifier.py
│
├── app.py
├── main.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Classifier

To run the command-line evaluation:

```bash
python main.py
```

Example output:

```text
                              text gold_status predicted_status      matched_cue  is_correct
0           The patient has asthma     ongoing          ongoing              has        True
1              Asthma better today     ongoing          ongoing           better        True
2  Asthma resolved after treatment    resolved         resolved         resolved        True
3                History of asthma    resolved         resolved       history of        True
4         No evidence of pneumonia     negated          negated  no evidence of        True

Accuracy: 100.0 %
```

The high accuracy is expected on this small demo dataset because the examples are designed to test the rule baseline directly. This should not be interpreted as production-level clinical performance.

---

## Run Tests

Run:

```bash
pytest
```

Expected output:

```text
5 passed
```

---

## Run the Streamlit App

Start the app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

Example input:

```text
Asthma better today
```

Example output:

```text
Status: ongoing
Matched cue: better
Reason: Ongoing/active cue found: 'better'
```

---

## Sample Dataset

The dataset is stored in:

```text
data/clinical_phrases.csv
```

Each row contains:

| Column | Description |
|---|---|
| `text` | Short clinical phrase |
| `condition` | Condition or symptom being evaluated |
| `gold_status` | Expected label |
| `explanation` | Human-readable explanation |

Example:

```csv
text,condition,gold_status,explanation
"The patient has asthma",asthma,ongoing,"Condition is currently present"
"Asthma better today",asthma,ongoing,"Better means improved but not resolved"
"Asthma resolved after treatment",asthma,resolved,"Explicit resolved cue"
"Patient denies chest pain",chest pain,negated,"Denies is a negation cue"
"Possible pneumonia",pneumonia,ambiguous,"Possible indicates uncertainty"
```

---

## Design Decisions

### Why rule-based?

A rule-based approach was chosen because the project is small, explainable, and easy to inspect. In clinical NLP, interpretability is important, especially when the system is making decisions about condition status.

This baseline makes it clear why a prediction was made by returning:

- predicted status
- matched cue
- explanation

Example:

```json
{
  "status": "ongoing",
  "cue": "better",
  "reason": "Ongoing/active cue found: 'better'"
}
```

---

### Why not full clinical NER?

For this demo, the condition is assumed to already be present in the short phrase or provided in the dataset.

A production system would likely add clinical Named Entity Recognition to first extract medical problems, symptoms, or diagnoses from longer notes.

Possible future pipeline:

```text
Clinical Note
    ↓
Condition / Symptom Extraction
    ↓
Context Window Around Each Entity
    ↓
Condition Status Classification
    ↓
Structured Output
```

---

### Why is “better” ongoing?

Words like `better`, `improving`, `stable`, and `controlled` usually indicate that the condition still exists but has changed in severity or control.

For example:

```text
Asthma better today
```

This does not mean asthma is gone. It means the asthma is improved. Therefore, the status is classified as `ongoing`.

---

## Limitations

This project is a baseline demo and is not intended for clinical use.

Current limitations:

- Small dataset
- Limited cue list
- No full clinical Named Entity Recognition
- No long-note context handling
- No section-aware logic
- No temporal reasoning beyond simple cues
- No handling of complex contradictions
- No machine learning model
- No clinical validation

Example of a challenging phrase:

```text
History of asthma, currently worsening
```

This contains both historical and active cues. A more advanced system would need better context and priority handling.

---

## Future Improvements

Possible improvements include:

1. Add more examples to the dataset.
2. Add section-aware classification, such as:
   - Past Medical History
   - Assessment and Plan
   - Review of Systems
3. Add clinical NER for condition extraction.
4. Add confidence scores.
5. Add error analysis.
6. Add support for longer clinical notes.
7. Add ML-based classification.
8. Add LLM-based classification with structured output.
9. Add evaluation metrics such as precision, recall, and F1-score.
10. Add more clinically realistic edge cases.

---


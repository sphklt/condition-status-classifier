"""
Note-level evaluation for the full pipeline.

Runs process_note() on each entry in data/annotated_notes.json and measures
precision, recall, and F1 against the expected condition→status annotations.

Matching strategy
-----------------
Each expected item specifies a 'keyword' (e.g. "chest pain") and a 'status'.
A pipeline result is considered a match if:
  - its condition text contains the keyword (case-insensitive), AND
  - its status equals the expected status.

Precision = TP / (TP + FP)   where FP = pipeline results that don't match any expected item
Recall    = TP / (TP + FN)   where FN = expected items not matched by any pipeline result
F1        = harmonic mean of precision and recall

Per-note and aggregate metrics are returned.
"""

import json
from pathlib import Path

from src.pipeline import process_note


def _match(condition_text: str, keyword: str) -> bool:
    return keyword.lower() in condition_text.lower()


def evaluate_notes(notes_path: str = "data/annotated_notes.json") -> dict:
    """
    Evaluate the pipeline on each annotated clinical note.

    Returns
    -------
    dict with keys:
        "notes"      — list of per-note result dicts
        "aggregate"  — overall precision, recall, F1, counts
    """
    notes_path = Path(notes_path)
    with open(notes_path) as f:
        annotated = json.load(f)

    note_results = []
    total_tp = total_fp = total_fn = 0

    for entry in annotated:
        pipeline_result = process_note(entry["note"])
        expected = entry["expected"]

        matched_expected: set[int] = set()   # indices of expected items that were found
        matched_pipeline: set[int] = set()   # indices of pipeline results that matched

        # Try to match each pipeline result to an expected item
        for p_idx, pred in enumerate(pipeline_result.conditions):
            for e_idx, exp in enumerate(expected):
                if e_idx in matched_expected:
                    continue
                if _match(pred.condition, exp["keyword"]) and pred.status == exp["status"]:
                    matched_expected.add(e_idx)
                    matched_pipeline.add(p_idx)
                    break

        tp = len(matched_expected)
        fp = len(pipeline_result.conditions) - len(matched_pipeline)
        fn = len(expected) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # Build per-item detail for display
        items = []
        for e_idx, exp in enumerate(expected):
            found = e_idx in matched_expected
            # Find which pipeline result matched (if any)
            matched_pred = None
            for p_idx, pred in enumerate(pipeline_result.conditions):
                if p_idx in matched_pipeline and _match(pred.condition, exp["keyword"]):
                    matched_pred = pred
                    break
            items.append({
                "keyword":         exp["keyword"],
                "expected_status": exp["status"],
                "found":           found,
                "predicted_condition": matched_pred.condition if matched_pred else None,
                "predicted_status":    matched_pred.status    if matched_pred else None,
                "predicted_section":   matched_pred.section   if matched_pred else None,
            })

        note_results.append({
            "id":        entry["id"],
            "title":     entry["title"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "items":     items,
            "pipeline_conditions": [
                {"condition": c.condition, "status": c.status,
                 "confidence": c.confidence, "section": c.section}
                for c in pipeline_result.conditions
            ],
        })

    # Aggregate
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = (2 * agg_precision * agg_recall / (agg_precision + agg_recall)
              if (agg_precision + agg_recall) > 0 else 0.0)

    return {
        "notes": note_results,
        "aggregate": {
            "precision": round(agg_precision, 3),
            "recall":    round(agg_recall, 3),
            "f1":        round(agg_f1, 3),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "n_notes": len(annotated),
        },
    }

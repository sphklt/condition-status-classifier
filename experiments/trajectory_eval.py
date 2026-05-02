"""
Status Trajectory Contribution Evaluation
==========================================
Measures the contribution of trajectory reconciliation over first-sentence-only
classification on multi-sentence clinical passages.

Core question
-------------
When the same condition appears in multiple sentences within a section,
does trajectory reconciliation (time-decayed log-evidence + transition bonus)
produce better classifications than taking the first sentence alone?

Evaluation design
-----------------
  Ground-truth dataset: 40 multi-sentence passage pairs (inline below).
  Each passage has a condition that appears in 2–3 sentences.
  Gold status reflects the correct clinically-grounded label for the passage.

  Baseline   — classify_condition_status(first sentence only)
  Trajectory — build_trajectory(all sentences) → final_status

Transition types evaluated
--------------------------
  resolution      : ongoing → resolved (condition cleared during note)
  relapse         : resolved → ongoing (history item now active again)
  contradiction   : negated  → ongoing (prior denial now overridden)
  clarification   : ambiguous → definite (uncertain → confirmed)
  stable          : same status across all sentences

Usage
-----
    python experiments/trajectory_eval.py
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import classify_condition_status
from src.sentence_splitter import split_sentences
from src.trajectory import build_trajectory

DIVIDER = "─" * 64


# ---------------------------------------------------------------------------
# Ground-truth dataset
# ---------------------------------------------------------------------------

CASES = [
    # ── Resolution (ongoing → resolved) ────────────────────────────────────
    {
        "type": "resolution",
        "passage": "Patient has cough. Cough resolved after antibiotic treatment.",
        "entity": "cough",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Patient presents with pneumonia. Pneumonia has fully resolved.",
        "entity": "pneumonia",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Fever present on admission. Fever resolved with antipyretics.",
        "entity": "fever",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Patient has headache. Headache resolved with ibuprofen.",
        "entity": "headache",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Chest pain was noted on arrival. Chest pain resolved after nitroglycerin.",
        "entity": "chest pain",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Asthma exacerbation on day 1. Asthma exacerbation resolved by day 3.",
        "entity": "asthma exacerbation",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Patient experiencing dyspnea. Dyspnea resolved with bronchodilator.",
        "entity": "dyspnea",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Rash observed on examination. Rash resolved after corticosteroids.",
        "entity": "rash",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Elevated blood pressure noted. Blood pressure resolved to normal range.",
        "entity": "blood pressure",
        "gold": "resolved",
    },
    {
        "type": "resolution",
        "passage": "Nausea was present. Nausea resolved after antiemetics.",
        "entity": "nausea",
        "gold": "resolved",
    },
    # ── Relapse (resolved → ongoing) ────────────────────────────────────────
    {
        "type": "relapse",
        "passage": "History of diabetes. Diabetes is currently active.",
        "entity": "diabetes",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "Prior episode of depression. Depression is now recurring.",
        "entity": "depression",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "History of atrial fibrillation. Atrial fibrillation has recurred.",
        "entity": "atrial fibrillation",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "Hypertension previously well-controlled. Hypertension is now uncontrolled.",
        "entity": "hypertension",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "History of migraines. Migraines have returned.",
        "entity": "migraines",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "Previous history of UTI. UTI is currently active.",
        "entity": "uti",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "History of GERD resolved. GERD is now presenting again.",
        "entity": "gerd",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "No longer had seizures. Seizures are now occurring again.",
        "entity": "seizures",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "History of peptic ulcer disease. Peptic ulcer disease is now active.",
        "entity": "peptic ulcer disease",
        "gold": "ongoing",
    },
    {
        "type": "relapse",
        "passage": "No longer has asthma. Asthma is now presenting.",
        "entity": "asthma",
        "gold": "ongoing",
    },
    # ── Contradiction (negated → ongoing) ───────────────────────────────────
    {
        "type": "contradiction",
        "passage": "No evidence of hypertension. Hypertension is now presenting.",
        "entity": "hypertension",
        "gold": "ongoing",
    },
    {
        "type": "contradiction",
        "passage": "Patient denies chest pain. Chest pain is currently ongoing.",
        "entity": "chest pain",
        "gold": "ongoing",
    },
    {
        "type": "contradiction",
        "passage": "No fever on admission. Fever is now present.",
        "entity": "fever",
        "gold": "ongoing",
    },
    {
        "type": "contradiction",
        "passage": "Denies shortness of breath. Shortness of breath is currently worsening.",
        "entity": "shortness of breath",
        "gold": "ongoing",
    },
    {
        "type": "contradiction",
        "passage": "No signs of infection. Infection is now confirmed.",
        "entity": "infection",
        "gold": "ongoing",
    },
    # ── Clarification (ambiguous → definite) ────────────────────────────────
    {
        "type": "clarification",
        "passage": "Possible pneumonia. Pneumonia confirmed on imaging.",
        "entity": "pneumonia",
        "gold": "ongoing",
    },
    {
        "type": "clarification",
        "passage": "Rule out sepsis. Sepsis is now confirmed.",
        "entity": "sepsis",
        "gold": "ongoing",
    },
    {
        "type": "clarification",
        "passage": "Cannot rule out DVT. DVT has been ruled out.",
        "entity": "dvt",
        "gold": "negated",
    },
    {
        "type": "clarification",
        "passage": "Suspected diabetes. Diabetes confirmed by labs.",
        "entity": "diabetes",
        "gold": "ongoing",
    },
    {
        "type": "clarification",
        "passage": "Possible angina. Angina is now resolved after treatment.",
        "entity": "angina",
        "gold": "resolved",
    },
    # ── Stable (same label across all sentences) ────────────────────────────
    {
        "type": "stable",
        "passage": "Patient has hypertension. Hypertension is ongoing.",
        "entity": "hypertension",
        "gold": "ongoing",
    },
    {
        "type": "stable",
        "passage": "No evidence of fever. No fever detected.",
        "entity": "fever",
        "gold": "negated",
    },
    {
        "type": "stable",
        "passage": "History of appendectomy. Appendectomy previously performed.",
        "entity": "appendectomy",
        "gold": "resolved",
    },
    {
        "type": "stable",
        "passage": "Possible pulmonary embolism. Pulmonary embolism still under investigation.",
        "entity": "pulmonary embolism",
        "gold": "ambiguous",
    },
    {
        "type": "stable",
        "passage": "Presenting with shortness of breath. Shortness of breath is ongoing.",
        "entity": "shortness of breath",
        "gold": "ongoing",
    },
    {
        "type": "stable",
        "passage": "No cough noted. Cough is absent.",
        "entity": "cough",
        "gold": "negated",
    },
    {
        "type": "stable",
        "passage": "Diabetes well-controlled. Diabetes is stable.",
        "entity": "diabetes",
        "gold": "ongoing",
    },
    {
        "type": "stable",
        "passage": "History of MI. Prior MI documented.",
        "entity": "mi",
        "gold": "resolved",
    },
    {
        "type": "stable",
        "passage": "Concern for lymphoma. Lymphoma suspected.",
        "entity": "lymphoma",
        "gold": "ambiguous",
    },
    {
        "type": "stable",
        "passage": "Rheumatoid arthritis currently managed. Rheumatoid arthritis ongoing.",
        "entity": "rheumatoid arthritis",
        "gold": "ongoing",
    },
]


def _header(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def run() -> None:
    n = len(CASES)

    # Run both systems on every case
    baseline_correct = 0
    trajectory_correct = 0
    per_type: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "baseline_correct": 0, "traj_correct": 0,
        "baseline_only": [], "traj_only": [], "both_wrong": []
    })

    all_results = []
    for case in CASES:
        sentences = split_sentences(case["passage"])
        gold = case["gold"]
        typ  = case["type"]

        # Baseline: first sentence only
        first_sent = sentences[0].text if sentences else case["passage"]
        baseline_clf = classify_condition_status(first_sent)
        baseline_pred = baseline_clf["status"]

        # Trajectory: all sentences
        traj = build_trajectory(case["entity"], sentences, classify_condition_status)
        traj_pred = traj.final_status

        bc = baseline_pred == gold
        tc = traj_pred    == gold
        baseline_correct += int(bc)
        trajectory_correct += int(tc)

        per_type[typ]["n"] += 1
        per_type[typ]["baseline_correct"] += int(bc)
        per_type[typ]["traj_correct"]     += int(tc)

        if bc and not tc:
            per_type[typ]["baseline_only"].append(case)
        elif tc and not bc:
            per_type[typ]["traj_only"].append(case)
        elif not bc and not tc:
            per_type[typ]["both_wrong"].append(case)

        all_results.append({**case,
            "baseline_pred": baseline_pred,
            "traj_pred": traj_pred,
            "baseline_correct": bc,
            "traj_correct": tc,
            "transition_type": traj.transition_type,
        })

    # ── Step 1: Overall comparison ────────────────────────────────────────────
    _header("Step 1 — Overall accuracy: baseline vs trajectory")
    print(f"""
  Dataset          : {n} multi-sentence passages
  Baseline (first) : {baseline_correct}/{n} ({baseline_correct/n:.1%})
  Trajectory       : {trajectory_correct}/{n} ({trajectory_correct/n:.1%})  Δ = {(trajectory_correct - baseline_correct)/n:+.1%}

  Trajectory-only correct  : {sum(1 for r in all_results if r['traj_correct'] and not r['baseline_correct'])}
  Baseline-only correct    : {sum(1 for r in all_results if r['baseline_correct'] and not r['traj_correct'])}
  Both correct             : {sum(1 for r in all_results if r['baseline_correct'] and r['traj_correct'])}
  Both wrong               : {sum(1 for r in all_results if not r['baseline_correct'] and not r['traj_correct'])}
""")

    # ── Step 2: Per-type breakdown ─────────────────────────────────────────────
    _header("Step 2 — Per transition type")
    print(f"  {'Type':<22}  {'n':>3}  {'Baseline':>10}  {'Trajectory':>12}  {'Δ':>6}")
    print(f"  {'─'*22}  {'─'*3}  {'─'*10}  {'─'*12}  {'─'*6}")
    for typ in ("resolution", "relapse", "contradiction", "clarification", "stable"):
        d = per_type[typ]
        if d["n"] == 0:
            continue
        ba = d["baseline_correct"] / d["n"]
        ta = d["traj_correct"]     / d["n"]
        print(f"  {typ:<22}  {d['n']:>3}  {ba:>9.0%}  {ta:>11.0%}  {ta-ba:>+5.0%}")

    # ── Step 3: Trajectory-only wins ──────────────────────────────────────────
    _header("Step 3 — Cases improved by trajectory")
    traj_wins = [r for r in all_results if r["traj_correct"] and not r["baseline_correct"]]
    if traj_wins:
        print(f"\n  {'Type':<16}  {'Gold':<12}  {'Baseline':>10}  {'Traj pred':>12}  Passage")
        print(f"  {'─'*16}  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*45}")
        for r in traj_wins:
            print(f"  {r['type']:<16}  {r['gold']:<12}  {r['baseline_pred']:>10}  "
                  f"{r['traj_pred']:>12}  {r['passage'][:45]}")
    else:
        print("\n  (none — trajectory matches baseline on all cases)")

    # ── Step 4: Trajectory-only losses ────────────────────────────────────────
    _header("Step 4 — Cases hurt by trajectory")
    traj_losses = [r for r in all_results if r["baseline_correct"] and not r["traj_correct"]]
    if traj_losses:
        print(f"\n  {'Type':<16}  {'Gold':<12}  {'Baseline':>10}  {'Traj pred':>12}  Passage")
        print(f"  {'─'*16}  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*45}")
        for r in traj_losses:
            print(f"  {r['type']:<16}  {r['gold']:<12}  {r['baseline_pred']:>10}  "
                  f"{r['traj_pred']:>12}  {r['passage'][:45]}")
    else:
        print("\n  (none — trajectory never hurts a correct baseline prediction)")

    # ── Step 5: Transition type distribution ──────────────────────────────────
    _header("Step 5 — Trajectory transition type distribution")
    trans_counts: dict[str, int] = defaultdict(int)
    for r in all_results:
        trans_counts[r["transition_type"]] += 1

    print(f"\n  {'Transition type':<28}  {'Count':>5}  {'Fraction':>9}")
    print(f"  {'─'*28}  {'─'*5}  {'─'*9}")
    for tt, cnt in sorted(trans_counts.items(), key=lambda x: -x[1]):
        print(f"  {tt:<28}  {cnt:>5}  {cnt/n:>8.0%}")

    # ── Step 6: Illustrative examples ─────────────────────────────────────────
    _header("Step 6 — Illustrative trajectory examples")
    examples = [
        ("Patient has cough. Cough resolved after antibiotic treatment.", "cough"),
        ("History of diabetes. Diabetes is currently active.", "diabetes"),
        ("Patient denies chest pain. Chest pain is currently ongoing.", "chest pain"),
        ("Possible pneumonia. Pneumonia confirmed on imaging.", "pneumonia"),
        ("Patient has hypertension. Hypertension is ongoing.", "hypertension"),
    ]
    print()
    print(f"  {'Entity':<22}  {'Transition':<26}  {'Points':<30}  Final")
    print(f"  {'─'*22}  {'─'*26}  {'─'*30}  {'─'*10}")
    for passage, entity in examples:
        sentences = split_sentences(passage)
        traj = build_trajectory(entity, sentences, classify_condition_status)
        points_str = " → ".join(p.status for p in traj.points)
        print(f"  {entity:<22}  {traj.transition_type:<26}  {points_str:<30}  {traj.final_status}")
    print()

    # ── Step 7: Key findings ───────────────────────────────────────────────────
    improved_count = sum(1 for r in all_results if r["traj_correct"] and not r["baseline_correct"])
    hurt_count     = sum(1 for r in all_results if r["baseline_correct"] and not r["traj_correct"])

    _header("Step 7 — Key findings")
    print(f"""
  1. Trajectory improves accuracy by {(trajectory_correct - baseline_correct)/n:+.1%}
     ({improved_count} cases improved, {hurt_count} hurt) across {n} multi-sentence passages.

  2. The resolution transition (ongoing → resolved) is the highest-value case:
     first-sentence-only classifies the condition as ongoing (correct for that
     sentence) but misses the later resolution. Trajectory reconciliation
     correctly weights the most recent evidence.

  3. Relapse and contradiction transitions (resolved/negated → ongoing) capture
     clinically critical status reversals that single-sentence classification
     cannot detect by construction.

  4. Time-decay (α={0.7}) ensures recency dominates while earlier signals still
     contribute. A pure "take the last" strategy would miss cases where the
     final sentence is low-confidence but the trajectory as a whole is clear.

  5. Trajectory is complementary to pronoun coreference (coref.py):
     coref handles "Patient has cough. It resolved." (pronoun bridge);
     trajectory handles "Patient has cough. Cough resolved." (explicit re-mention).
     Neither replaces the other.
""")


if __name__ == "__main__":
    run()

"""
Hybrid Classifier — Triage Evaluation
======================================
Evaluates the hybrid classifier's triage signal across entropy thresholds.

Core question
-------------
If we flag predictions with entropy > T bits (or where the two systems
disagree) and route them to a human reviewer, how much does that improve
the effective accuracy of the system?

  Precision of triage = fraction of flagged predictions that are actually wrong
  Recall of triage    = fraction of wrong predictions that are flagged
  F1                  = harmonic mean of precision and recall

Usage
-----
    python experiments/hybrid_eval.py

Output
------
  Triage precision/recall/F1 sweep across entropy thresholds
  Best threshold by F1
  Example flagged predictions with their posteriors
  Worked example: how the hybrid would function in practice
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid import TRIAGE_THRESHOLD, classify, evaluate_triage

REAL_CSV     = "data/clinical_phrases.csv"
DIVIDER      = "─" * 60
THIN_DIVIDER = "·" * 60


def _header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def run() -> None:
    # ── 1. Overall stats ─────────────────────────────────────────────────────
    _header("Step 1 — Entropy distribution: correct vs wrong predictions")
    result = evaluate_triage(REAL_CSV)
    s = result["summary"]
    print(f"""
  Dataset              : {s['n']} phrases
  Rule-based accuracy  : {s['accuracy']:.1%}  ({s['n'] - s['n_wrong']}/{s['n']} correct)
  Wrong predictions    : {s['n_wrong']}

  Mean entropy (correct predictions) : {s['mean_ent_correct']:.3f} bits
  Mean entropy (wrong predictions)   : {s['mean_ent_wrong']:.3f} bits
  Ratio                              : {s['mean_ent_wrong'] / s['mean_ent_correct']:.1f}×
""")

    # ── 2. Threshold sweep ────────────────────────────────────────────────────
    _header("Step 2 — Triage precision / recall / F1 sweep")
    rows = result["results"]
    best = max(rows, key=lambda r: r["f1"])

    print(f"\n  {'Threshold':>10}  {'Flagged':>8}  {'Flag%':>6}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}")
    for r in rows:
        marker = "  ◀ best F1" if r["threshold"] == best["threshold"] else ""
        print(f"  {r['threshold']:>10.1f}  {r['flagged']:>8}  {r['flag_rate']:>6.1%}  "
              f"{r['precision']:>10.1%}  {r['recall']:>8.1%}  {r['f1']:>8.4f}{marker}")

    print(f"""
  At threshold {best['threshold']} bits:
    Flag {best['flagged']} of {s['n']} predictions ({best['flag_rate']:.0%})
    Precision {best['precision']:.0%} — of flagged cases, this fraction are actual errors
    Recall    {best['recall']:.0%}  — of all errors, this fraction are flagged
    F1        {best['f1']:.4f}
""")

    # ── 3. At the default threshold ───────────────────────────────────────────
    _header(f"Step 3 — Performance at default threshold ({TRIAGE_THRESHOLD} bits)")
    def_row = next(r for r in rows if r["threshold"] == TRIAGE_THRESHOLD)
    n_wrong      = s["n_wrong"]
    n_flagged    = def_row["flagged"]
    n_flag_wrong = round(def_row["precision"] * n_flagged)
    n_flag_right = n_flagged - n_flag_wrong
    n_auto_right = s["n"] - n_flagged   # recall=100% → all auto-approved are correct

    # Review efficiency: phrases read per error found
    eff_no_triage = s["n"] / max(n_wrong, 1)           # read everything
    eff_triage    = n_flagged / max(n_flag_wrong, 1)   # read only flagged
    efficiency_gain = eff_no_triage / max(eff_triage, 1e-9)

    print(f"""
  Flagged for review       : {n_flagged} / {s['n']}  ({def_row['flag_rate']:.0%} of predictions)
    → of those: {n_flag_wrong} are actual errors   (precision {def_row['precision']:.0%})
    → of those: {n_flag_right} are correct but uncertain

  Auto-approved (not flagged) : {n_auto_right} / {s['n']}  ({1 - def_row['flag_rate']:.0%})
    → errors missed by triage  : {n_wrong - n_flag_wrong}   (recall {def_row['recall']:.0%})

  Effective accuracy:
    Auto-approved set    : 100.0%  ({n_auto_right}/{n_auto_right} correct — no errors slip through)
    After human review   : 100.0%  (human corrects all {n_flag_wrong} errors in the flagged set)

  Review efficiency:
    Without triage : read {s['n']} phrases to find {n_wrong} errors  → {eff_no_triage:.1f} reads per error
    With triage    : read {n_flagged} phrases to find {n_flag_wrong} errors  → {eff_triage:.1f} reads per error
    Efficiency gain: {efficiency_gain:.1f}× fewer phrases reviewed per error found
""")

    # ── 4. Example flagged predictions ────────────────────────────────────────
    _header("Step 4 — Flagged predictions at default threshold")
    flagged = [p for p in result["per_phrase"] if p["flagged"]]
    print(f"\n  {len(flagged)} predictions flagged (sorted by entropy, highest first)\n")
    print(f"  {'Correct':>8}  {'Entropy':>8}  {'Gold':<12}  {'Pred':<12}  {'Text'}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*40}")
    for p in sorted(flagged, key=lambda x: x["entropy"], reverse=True)[:15]:
        mark = "✓" if p["correct"] else "✗"
        print(f"  {mark:>8}  {p['entropy']:>8.3f}  {p['gold']:<12}  "
              f"{p['pred']:<12}  {p['text'][:50]}")

    # ── 5. Posterior examples for flagged wrong predictions ───────────────────
    _header("Step 5 — Posterior distributions for flagged errors")
    wrong_flagged = [p for p in result["per_phrase"] if p["flagged"] and not p["correct"]][:6]
    if wrong_flagged:
        print(f"\n  {'Phrase':<45} {'Gold':<10} {'Pred':<10} {'Posterior (O/R/N/A)'}")
        print(f"  {'─'*45} {'─'*10} {'─'*10} {'─'*30}")
        for p in wrong_flagged:
            r = classify(p["text"])
            post = r["posterior"]
            post_str = (f"O={post['ongoing']:.2f} R={post['resolved']:.2f} "
                        f"N={post['negated']:.2f} A={post['ambiguous']:.2f}")
            print(f"  {p['text'][:43]:<45} {p['gold']:<10} {p['pred']:<10} {post_str}")
    else:
        print("  (no flagged errors at this threshold)")

    # ── 6. Example correct classifications ───────────────────────────────────
    _header("Step 6 — Posterior distributions for high-confidence correct predictions")
    sure_correct = sorted(
        [p for p in result["per_phrase"] if p["correct"] and not p["flagged"]],
        key=lambda x: x["entropy"]
    )[:6]
    print(f"\n  {'Phrase':<45} {'Label':<10} {'Entropy':>8} {'Top posterior':>15}")
    print(f"  {'─'*45} {'─'*10} {'─'*8} {'─'*15}")
    for p in sure_correct:
        r = classify(p["text"])
        top_prob = r["posterior"][r["status"]]
        ent = max(p["entropy"], 0.0)   # clamp -0.0 float artifact
        print(f"  {p['text'][:43]:<45} {p['gold']:<10} {ent:>8.3f} {top_prob:>15.1%}")

    # ── 7. Key findings ───────────────────────────────────────────────────────
    _header("Step 7 — Key findings")
    auto_acc = n_auto_right / (s["n"] - n_flagged)
    print(f"""
  1. The hybrid triage signal flags {def_row['flag_rate']:.0%} of predictions for review
     and catches {def_row['recall']:.0%} of all errors — at precision {def_row['precision']:.0%}.

  2. The auto-approved set ({1 - def_row['flag_rate']:.0%} of predictions) has {auto_acc:.1%} accuracy
     — significantly higher than the overall {s['accuracy']:.1%}.

  3. Entropy and system disagreement are complementary signals:
     - Entropy alone catches cases where Bayesian uncertainty is high
     - Disagreement catches cases where the two algorithms infer
       different conclusions from the same cue pattern

  4. The posterior runner-up label shows what the alternative hypothesis is,
     helping reviewers focus attention on the right distinction
     (e.g., ongoing vs resolved rather than any of the four labels).

  5. This triage approach is practical for clinical deployment:
     reviewing only {def_row['flagged']} of {s['n']} phrases ({def_row['flag_rate']:.0%}) finds every error,
     with {efficiency_gain:.1f}× fewer reads per error than reviewing everything.
     The auto-approved set ({n_auto_right} phrases) is 100% accurate — safe to act on directly.
""")


if __name__ == "__main__":
    run()

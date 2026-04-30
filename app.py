import streamlit as st
import pandas as pd

from src.classifier import classify_condition_status
from src.utils import evaluate_dataset
from src.pipeline import process_note, format_results
from src.ner import active_ner_method
from src.calibration import reliability_diagram
from src.note_evaluator import evaluate_notes

st.set_page_config(
    page_title="Clinical Condition Status Classifier",
    page_icon="🩺",
    layout="centered",
)

st.title("Clinical Condition Status Classifier")

STATUS_COLOURS = {
    "ongoing":   "#2ecc71",
    "resolved":  "#3498db",
    "negated":   "#e74c3c",
    "ambiguous": "#f39c12",
}

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_phrase, tab_note, tab_eval = st.tabs([
    "Single Phrase", "Full Clinical Note", "Evaluate Dataset"
])


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Single phrase classifier
# ════════════════════════════════════════════════════════════════════════════
with tab_phrase:
    st.subheader("Classify a clinical phrase")
    st.caption(
        "Classifies a short clinical phrase into **ongoing**, **resolved**, "
        "**negated**, or **ambiguous**."
    )

    examples = [
        "Asthma better today",
        "h/o diabetes",
        "Patient has no fever",
        "No active infection",
        "No longer has headache",
        "History of asthma, currently worsening",
        "Fever -ve",
        "s/p appendectomy",
        "Concern for pulmonary embolism",
        "DM diagnosed 3 years ago",
        "I had severe flu which I think is getting better now. But after a couple of days, it got completely over.",
    ]

    selected = st.selectbox(
        "Load an example (or type your own below)",
        ["— type your own —"] + examples,
        key="phrase_example",
    )

    # Streamlit ignores value= on re-render if the key is already in session_state.
    # Explicitly update session_state so the text area reflects the chosen example.
    if selected != "— type your own —":
        st.session_state["phrase_input"] = selected

    user_text = st.text_area("Clinical phrase", height=80, key="phrase_input")

    if st.button("Classify", type="primary", key="btn_phrase"):
        if not user_text.strip():
            st.warning("Please enter a clinical phrase.")
        else:
            result = classify_condition_status(user_text)
            colour = STATUS_COLOURS.get(result["status"], "#888")

            st.markdown("### Prediction")
            cal_conf = result.get("calibrated_confidence", result["confidence"])
            st.markdown(
                f"<span style='font-size:1.4rem; font-weight:bold; color:{colour};'>"
                f"{result['status'].upper()}</span> &nbsp; "
                f"<span style='color:#888'>calibrated confidence: {cal_conf:.0%}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Raw cue score: {result['confidence']:.0%} → "
                f"calibrated probability of being correct: {cal_conf:.0%}"
            )
            st.write(f"**Key signal:** `{result['cue']}`")
            st.write(f"**Reason:** {result['reason']}")

            with st.expander("Signal breakdown"):
                sigs = result["signals"]
                score_df = pd.DataFrame(
                    [{"category": k, "score": round(v, 3)}
                     for k, v in sigs.items() if isinstance(v, float)]
                ).sort_values("score", ascending=False)
                st.dataframe(score_df, hide_index=True, use_container_width=True)
                if sigs.get("abbreviations"):
                    st.write("**Abbreviations expanded:**", ", ".join(sigs["abbreviations"]))
                if sigs.get("pseudo_negations"):
                    st.write("**Pseudo-negations masked:**", ", ".join(sigs["pseudo_negations"]))
                if sigs.get("temporal") != "none":
                    st.write(f"**Temporal signal:** {sigs['temporal']}")
                if sigs.get("clause_used") == "final_clause":
                    st.info("Classification based on the **final clause** of a multi-part sentence.")


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Full clinical note pipeline
# ════════════════════════════════════════════════════════════════════════════
with tab_note:
    st.subheader("Process a full clinical note")
    st.caption(
        "Runs the full pipeline: **section detection → NER → context extraction → classification**. "
        "Each detected condition is classified in context, with the note section used as a tiebreaker."
    )

    ner_badge = "🔬 SciSpaCy" if active_ner_method() == "scispacy" else "📋 Vocabulary fallback"
    st.info(f"NER method: {ner_badge}")

    sample_note = """\
Chief Complaint:
Shortness of breath and fatigue for 3 days.

History of Present Illness:
67-year-old female presenting with worsening dyspnea. She reports progressive
shortness of breath over the past 3 days. Denies chest pain or fever.

Past Medical History:
Hypertension, type 2 diabetes mellitus (diagnosed 5 years ago), h/o pneumonia
(resolved last year), atrial fibrillation controlled on medication.

Past Surgical History:
s/p appendectomy 10 years ago.

Medications:
Metformin, lisinopril, warfarin.

Review of Systems:
Positive for dyspnea and fatigue. Negative for chest pain, fever, or cough.

Assessment:
Possible heart failure exacerbation. Rule out pulmonary embolism.
Hypertension well-controlled. Diabetes stable.
"""

    note_text = st.text_area(
        "Paste clinical note here",
        value=sample_note,
        height=300,
        key="note_input",
    )

    if st.button("Process Note", type="primary", key="btn_note"):
        if not note_text.strip():
            st.warning("Please enter a clinical note.")
        else:
            with st.spinner("Running pipeline…"):
                pipeline_result = process_note(note_text)

            st.markdown("### Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Conditions found", len(pipeline_result.conditions))
            col2.metric("Sections detected", len(pipeline_result.sections_found))
            col3.metric("NER method", pipeline_result.ner_method)

            if pipeline_result.sections_found:
                st.write("**Sections detected:**", " · ".join(pipeline_result.sections_found))

            if pipeline_result.conditions:
                rows = []
                for r in pipeline_result.conditions:
                    rows.append({
                        "condition": r.condition,
                        "status": r.status,
                        "confidence": f"{r.confidence:.0%}",
                        "section": r.section,
                        "prior override": "yes" if r.overridden_by_prior else "",
                    })
                df = pd.DataFrame(rows)

                # Colour-code the status column
                def highlight_status(val):
                    colour = STATUS_COLOURS.get(val, "#888")
                    return f"color: {colour}; font-weight: bold"

                st.dataframe(
                    df.style.map(highlight_status, subset=["status"]),
                    hide_index=True,
                    use_container_width=True,
                )

                with st.expander("Reasoning details"):
                    for r in pipeline_result.conditions:
                        st.markdown(f"**{r.condition}** → `{r.status}`")
                        st.caption(f"Section: {r.section} | {r.reason}")
                        st.caption(f"Context: _{r.context[:120]}…_")
                        st.divider()

            if pipeline_result.warnings:
                with st.expander("Warnings"):
                    for w in pipeline_result.warnings:
                        st.warning(w)


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Dataset evaluation
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("Evaluate sample dataset")
    st.caption("Runs the phrase classifier over the labelled CSV dataset.")

    # ── Phrase-level evaluation ───────────────────────────────────────────────
    if st.button("Run phrase evaluation", key="btn_eval"):
        df = evaluate_dataset("data/clinical_phrases.csv")
        accuracy = df["is_correct"].mean()

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy:.0%}")
        col2.metric("Correct / Total", f"{df['is_correct'].sum()} / {len(df)}")

        st.dataframe(
            df[["text", "condition", "gold_status", "predicted_status",
                "confidence", "matched_cue", "is_correct"]],
            hide_index=True,
            use_container_width=True,
        )

        wrong = df[~df["is_correct"]]
        if not wrong.empty:
            st.warning(f"{len(wrong)} misclassified phrase(s):")
            for _, row in wrong.iterrows():
                st.write(
                    f"- **{row['text']}** → predicted `{row['predicted_status']}` "
                    f"(expected `{row['gold_status']}`): {row['prediction_reason']}"
                )

    st.divider()

    # ── Calibration analysis ─────────────────────────────────────────────────
    st.subheader("Confidence calibration")
    st.caption(
        "A well-calibrated model predicts confidence X% and is correct X% of the time. "
        "The reliability diagram shows actual accuracy vs. predicted confidence per bin."
    )

    if st.button("Run calibration analysis", key="btn_cal"):
        cal_df = reliability_diagram("data/clinical_phrases.csv")
        ece = cal_df.attrs["ece"]
        n_correct = cal_df.attrs["n_correct"]
        n_total = cal_df.attrs["n_total"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Overall accuracy", f"{n_correct / n_total:.0%}")
        col2.metric("ECE", f"{ece:.3f}", help="Expected Calibration Error — lower is better. 0 = perfect.")
        col3.metric("Phrases evaluated", str(n_total))

        filled = cal_df[cal_df["count"] > 0].copy()
        filled["bin_label"] = filled["bin_lower"].apply(lambda x: f"{x:.1f}") + "–" + filled["bin_upper"].apply(lambda x: f"{x:.1f}")

        chart_df = filled[["bin_label", "accuracy", "avg_confidence", "count"]].rename(
            columns={"accuracy": "Actual accuracy", "avg_confidence": "Avg confidence"}
        )
        st.bar_chart(chart_df.set_index("bin_label")[["Actual accuracy", "Avg confidence"]])

        st.dataframe(
            filled[["bin_label", "count", "avg_confidence", "accuracy", "gap"]].rename(
                columns={
                    "bin_label": "confidence bin", "avg_confidence": "mean confidence",
                    "gap": "calibration gap"
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        if ece < 0.05:
            st.success(f"ECE = {ece:.3f} — well-calibrated. Confidence scores track actual accuracy closely.")
        elif ece < 0.10:
            st.info(f"ECE = {ece:.3f} — moderate calibration. Some bins over- or under-estimate accuracy.")
        else:
            st.warning(f"ECE = {ece:.3f} — notable miscalibration. Consider fitting a Platt scaler on a larger dataset.")

        st.caption(
            "Note: calibration estimated on 39 phrases. A statistically robust calibration "
            "curve requires ≥500 labelled examples."
        )

    st.divider()

    # ── Note-level evaluation ────────────────────────────────────────────────
    st.subheader("Full note pipeline evaluation")
    st.caption(
        "Runs the full pipeline on 4 annotated clinical notes and reports "
        "**precision / recall / F1** per note and in aggregate."
    )

    if st.button("Run note evaluation", key="btn_note_eval"):
        with st.spinner("Processing 4 clinical notes…"):
            note_eval = evaluate_notes("data/annotated_notes.json")

        agg = note_eval["aggregate"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precision", f"{agg['precision']:.0%}")
        col2.metric("Recall",    f"{agg['recall']:.0%}")
        col3.metric("F1",        f"{agg['f1']:.0%}")
        col4.metric("Notes evaluated", str(agg["n_notes"]))

        st.write(f"TP: **{agg['tp']}** · FP: **{agg['fp']}** · FN: **{agg['fn']}**")

        for note in note_eval["notes"]:
            with st.expander(f"{note['title']} — F1 {note['f1']:.0%}  (P {note['precision']:.0%} / R {note['recall']:.0%})"):
                rows = []
                for item in note["items"]:
                    rows.append({
                        "expected keyword":    item["keyword"],
                        "expected status":     item["expected_status"],
                        "found":               "✓" if item["found"] else "✗",
                        "predicted condition": item["predicted_condition"] or "—",
                        "predicted status":    item["predicted_status"] or "—",
                        "section":             item["predicted_section"] or "—",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                fp_conditions = [
                    c for c in note["pipeline_conditions"]
                    if not any(
                        item["keyword"].lower() in c["condition"].lower()
                        for item in note["items"]
                    )
                ]
                if fp_conditions:
                    st.caption(
                        "Pipeline also found (not in annotations): "
                        + ", ".join(f"`{c['condition']}` ({c['status']})" for c in fp_conditions)
                    )

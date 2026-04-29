import streamlit as st
import pandas as pd

from src.classifier import classify_condition_status
from src.utils import evaluate_dataset
from src.pipeline import process_note, format_results
from src.ner import active_ner_method

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
            st.markdown(
                f"<span style='font-size:1.4rem; font-weight:bold; color:{colour};'>"
                f"{result['status'].upper()}</span> &nbsp; "
                f"<span style='color:#888'>confidence: {result['confidence']:.0%}</span>",
                unsafe_allow_html=True,
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

    if st.button("Run evaluation", key="btn_eval"):
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

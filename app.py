import streamlit as st
import pandas as pd

from src.classifier import classify_condition_status
from src.utils import evaluate_dataset


st.set_page_config(
    page_title="Clinical Condition Status Classifier",
    page_icon="🩺",
    layout="centered",
)

st.title("Clinical Condition Status Classifier")
st.write(
    """
    This demo classifies short clinical phrases into:
    **ongoing**, **resolved**, **negated**, or **ambiguous**.
    """
)

st.subheader("Try a clinical phrase")

user_text = st.text_area(
    "Enter clinical phrase",
    value="Asthma better today",
    height=100,
)

if st.button("Classify"):
    result = classify_condition_status(user_text)

    st.markdown("### Prediction")
    st.write(f"**Status:** `{result['status']}`")
    st.write(f"**Matched cue:** `{result['cue']}`")
    st.write(f"**Reason:** {result['reason']}")

st.divider()

st.subheader("Evaluate sample dataset")

if st.button("Run evaluation"):
    df = evaluate_dataset("data/clinical_phrases.csv")

    accuracy = df["is_correct"].mean()

    st.write(f"**Accuracy on sample dataset:** {round(accuracy * 100, 2)}%")

    st.dataframe(
        df[
            [
                "text",
                "condition",
                "gold_status",
                "predicted_status",
                "matched_cue",
                "is_correct",
            ]
        ]
    )
import pandas as pd
from src.classifier import classify_condition_status


def evaluate_dataset(csv_path: str) -> pd.DataFrame:
    """
    Reads a CSV file with clinical phrases and gold labels,
    runs the classifier, and returns a dataframe with predictions.
    """

    df = pd.read_csv(csv_path)

    predictions = []
    cues = []
    reasons = []

    for text in df["text"]:
        result = classify_condition_status(text)
        predictions.append(result["status"])
        cues.append(result["cue"])
        reasons.append(result["reason"])

    df["predicted_status"] = predictions
    df["matched_cue"] = cues
    df["prediction_reason"] = reasons
    df["is_correct"] = df["gold_status"] == df["predicted_status"]

    return df
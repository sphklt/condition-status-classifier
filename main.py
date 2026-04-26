from src.utils import evaluate_dataset


def main():
    result_df = evaluate_dataset("data/clinical_phrases.csv")

    print(result_df[[
        "text",
        "gold_status",
        "predicted_status",
        "matched_cue",
        "is_correct"
    ]])

    accuracy = result_df["is_correct"].mean()
    print("\nAccuracy:", round(accuracy * 100, 2), "%")


if __name__ == "__main__":
    main()
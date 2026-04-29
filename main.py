from src.utils import evaluate_dataset


def main():
    result_df = evaluate_dataset("data/clinical_phrases.csv")

    print(result_df[[
        "text",
        "gold_status",
        "predicted_status",
        "matched_cue",
        "confidence",
        "is_correct",
    ]].to_string(index=False))

    accuracy = result_df["is_correct"].mean()
    print(f"\nAccuracy: {round(accuracy * 100, 2)} %  ({result_df['is_correct'].sum()}/{len(result_df)} correct)")

    wrong = result_df[~result_df["is_correct"]]
    if not wrong.empty:
        print("\nMisclassified:")
        for _, row in wrong.iterrows():
            print(f"  [{row['gold_status']} → {row['predicted_status']}]  {row['text']}")
            print(f"    reason: {row['prediction_reason']}")


if __name__ == "__main__":
    main()

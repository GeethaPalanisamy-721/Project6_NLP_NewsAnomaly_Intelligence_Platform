"""
final_anomaly_score.py
----------------------
Purpose:
Combine multiple anomaly signals into ONE final decision label
+ clean noisy content locations systematically.

Signals Used:
1️⃣ Linguistic anomaly
2️⃣ Location anomaly
3️⃣ Temporal anomaly

Final Labels:
NORMAL    → No anomalies
REVIEW    → One anomaly signal
RED FLAG  → Two or more anomaly signals

Output:
data/processed/final_anomaly_results.csv
"""

import pandas as pd
from pathlib import Path

# ✅ import location cleaner 
from src.features.location_cleaning import clean_location


def main():
    print("🚨 Computing final anomaly labels with clean locations...")

    # --------------------------------------------------
    # 1️⃣ Load full feature set
    # --------------------------------------------------
    input_path = Path("data/processed/full_feature_set.csv")
    df = pd.read_csv(input_path)

    print(f"✔ Articles loaded: {len(df)}")

    # --------------------------------------------------
    # 2️⃣ Clean noisy content locations
    # --------------------------------------------------
    df[["location_clean", "location_type"]] = (
        df["content_location"]
        .apply(lambda x: pd.Series(clean_location(x)))
    )

    print("✔ Content locations cleaned")

    # --------------------------------------------------
    # 3️⃣ Convert anomaly signals to numeric flags
    # --------------------------------------------------
    df["linguistic_flag"] = df["is_anomaly"].map({
        "Anomaly": 1,
        "Normal": 0
    })

    df["location_flag"] = df["location_anomaly"].map({
        "Anomaly": 1,
        "Normal": 0,
        "Review": 0   # conservative handling
    })

    df["temporal_flag"] = df["temporal_anomaly"].map({
        "Anomaly": 1,
        "Normal": 0
    })

    # --------------------------------------------------
    # 4️⃣ Total anomaly score
    # --------------------------------------------------
    df["total_anomaly_score"] = (
        df["linguistic_flag"] +
        df["location_flag"] +
        df["temporal_flag"]
    )

    # --------------------------------------------------
    # 5️⃣ Final label assignment
    # --------------------------------------------------
    def assign_final_label(score):
        if score == 0:
            return "NORMAL"
        elif score == 1:
            return "REVIEW"
        else:
            return "RED FLAG"

    df["final_label"] = df["total_anomaly_score"].apply(assign_final_label)

    # --------------------------------------------------
    # 6️⃣ Save final results
    # --------------------------------------------------
    output_path = Path("data/processed/final_anomaly_results.csv")
    df.to_csv(output_path, index=False)

    print("✅ Final anomaly labeling completed")
    print(df["final_label"].value_counts())
    print("📁 Saved to:", output_path)


if __name__ == "__main__":
    main()

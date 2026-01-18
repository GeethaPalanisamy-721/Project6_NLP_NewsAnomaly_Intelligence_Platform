"""
Temporal Feature Extraction
---------------------------
Extracts time-based features from Date column
for temporal anomaly detection.

Output:
data/processed/news_with_temporal_features.csv
"""

import pandas as pd
from pathlib import Path


def main():
    print("⏰ Extracting temporal features...")

    # 1️⃣ Load data
    df = pd.read_csv("data/processed/news_with_sentiment.csv")

    # 2️⃣ Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 3️⃣ Extract temporal features
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["weekday"] = df["Date"].dt.weekday        # 0 = Monday
    df["weekday_name"] = df["Date"].dt.day_name()

    # 4️⃣ Save output
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "news_with_temporal_features.csv"
    df.to_csv(output_path, index=False)

    print("✅ Temporal features extracted successfully")
    print(df[["Date", "year", "month", "day", "weekday_name"]].head())


if __name__ == "__main__":
    main()

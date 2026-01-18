"""
Temporal Anomaly Detection
--------------------------
Detects sudden spikes in news volume over time
using rolling statistics (mean + std).
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("⏳ Running temporal anomaly detection...")

    # 1️⃣ Load data
    df = pd.read_csv("data/processed/news_with_temporal_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # 2️⃣ Aggregate article count per day
    daily_counts = (
        df.groupby("Date")
          .size()
          .reset_index(name="article_count")
          .sort_values("Date")
    )

    # 3️⃣ Rolling statistics (7-day window)
    daily_counts["rolling_mean"] = (
        daily_counts["article_count"]
        .rolling(window=7, min_periods=3)
        .mean()
    )

    daily_counts["rolling_std"] = (
        daily_counts["article_count"]
        .rolling(window=7, min_periods=3)
        .std()
    )

    # 4️⃣ Z-score calculation
    daily_counts["z_score"] = (
        (daily_counts["article_count"] - daily_counts["rolling_mean"]) /
        daily_counts["rolling_std"]
    )

    # 5️⃣ Temporal anomaly flag
    daily_counts["temporal_anomaly"] = np.where(
        daily_counts["z_score"] > 1.8,
        "Anomaly",
        "Normal"
    )

    # 6️⃣ Merge back to original data
    df = df.merge(
        daily_counts[["Date", "temporal_anomaly"]],
        on="Date",
        how="left"
    )

    # 7️⃣ Save output
    output_path = Path("data/processed/news_with_temporal_anomaly.csv")
    df.to_csv(output_path, index=False)

    print("✅ Temporal anomaly detection completed")
    print(df["temporal_anomaly"].value_counts())


if __name__ == "__main__":
    main()

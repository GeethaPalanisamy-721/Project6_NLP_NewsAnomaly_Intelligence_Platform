"""
feature_union.py
----------------
Purpose:
Merge all engineered features into ONE final dataset
WITHOUT increasing row count.

Key Principles:
✔ article_id is the single source of truth
✔ One row = One article
✔ INNER JOIN everywhere (safe joins)

Output:
data/processed/full_feature_set.csv
"""

import pandas as pd
from pathlib import Path


def main():
    print("🔗 Starting safe feature union...")

    base_path = Path("data/processed")

    # --------------------------------------------------
    # 1️⃣ Load all processed feature files
    # --------------------------------------------------
    cleaned_df = pd.read_csv(base_path / "news_cleaned.csv")
    location_df = pd.read_csv(base_path / "news_with_location.csv")
    sentiment_df = pd.read_csv(base_path / "news_with_sentiment.csv")
    topic_df = pd.read_csv(base_path / "news_with_topics.csv")
    temporal_df = pd.read_csv(base_path / "news_with_temporal_features.csv")
    linguistic_df = pd.read_csv(base_path / "anomaly_scores.csv")
    temporal_anomaly_df = pd.read_csv(base_path / "news_with_temporal_anomaly.csv")

    print(f"✔ Base articles loaded: {len(cleaned_df)}")

    # --------------------------------------------------
    # 2️⃣ Start merging (article_id ONLY)
    # --------------------------------------------------
    df = cleaned_df.copy()

    # -------- Location (claim vs content + anomaly) --------
    df = df.merge(
        location_df[
            [
                "article_id",
                "claimed_location",
                "content_location",
                "location_anomaly"
            ]
        ],
        on="article_id",
        how="inner"
    )

    # -------- Sentiment --------
    df = df.merge(
        sentiment_df[
            [
                "article_id",
                "sentiment_positive",
                "sentiment_negative",
                "sentiment_neutral",
                "sentiment_label"
            ]
        ],
        on="article_id",
        how="inner"
    )

    # -------- Topics + Keywords --------
    df = df.merge(
        topic_df[
            [
                "article_id",
                "topic_id",
                "topic_keywords"
            ]
        ],
        on="article_id",
        how="inner"
    )

    # -------- Temporal Features --------
    df = df.merge(
        temporal_df[
            [
                "article_id",
                "year",
                "month",
                "day",
                "weekday_name"
            ]
        ],
        on="article_id",
        how="inner"
    )

    # -------- Linguistic Anomaly --------
    df = df.merge(
        linguistic_df[
            [
                "article_id",
                "is_anomaly"
            ]
        ],
        on="article_id",
        how="inner"
    )

    # -------- Temporal Anomaly --------
    df = df.merge(
        temporal_anomaly_df[
            [
                "article_id",
                "temporal_anomaly"
            ]
        ],
        on="article_id",
        how="inner"
    )

    # --------------------------------------------------
    # 3️⃣ Final sanity check
    # --------------------------------------------------
    print(f"✅ Rows after merge: {len(df)}")

    if len(df) != len(cleaned_df):
        raise ValueError("❌ Row count mismatch! Feature union created extra rows.")

    # --------------------------------------------------
    # 4️⃣ Save final dataset
    # --------------------------------------------------
    output_path = base_path / "full_feature_set.csv"
    df.to_csv(output_path, index=False)

    print("🎉 Feature union completed successfully")
    print("📁 Saved to:", output_path)


if __name__ == "__main__":
    main()

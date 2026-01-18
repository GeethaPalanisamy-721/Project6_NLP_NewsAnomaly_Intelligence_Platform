"""
topic_keywords.py
-----------------
Purpose:
--------
Create a topic-level lookup table so dashboards
can show human-readable topic names.

Input:
------
data/processed/news_with_topics.csv

Output:
-------
data/processed/topic_keywords.csv
"""

import pandas as pd
from pathlib import Path


def main():
    print("🧩 Creating topic keyword lookup table...")

    input_path = Path("data/processed/news_with_topics.csv")
    output_path = Path("data/processed/topic_keywords.csv")

    df = pd.read_csv(input_path)

    # Safety check
    required_cols = {"topic_id", "topic_keywords"}
    if not required_cols.issubset(df.columns):
        raise ValueError("topic_id or topic_keywords missing")

    # -------------------------------
    # One row per topic
    # -------------------------------
    topic_df = (
        df[["topic_id", "topic_keywords"]]
        .drop_duplicates()
        .sort_values("topic_id")
        .reset_index(drop=True)
    )

    # Optional: rename for clarity
    topic_df.rename(columns={"topic_keywords": "keywords"}, inplace=True)

    # Save
    topic_df.to_csv(output_path, index=False)

    print("✅ topic_keywords.csv created")
    print(topic_df.head())


if __name__ == "__main__":
    main()

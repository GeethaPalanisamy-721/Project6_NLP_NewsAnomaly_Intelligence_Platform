"""
brand_risk.py
--------------
Purpose:
--------
Compute composite Brand / Organization risk scores from news articles.

Key Idea:
---------
Brand risk should reflect BOTH:
1️⃣ Severity of risk signals
2️⃣ Frequency of risky mentions

This avoids false alarms from single-article brands.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    print("🏷️ Computing composite brand risk scores...")

    # --------------------------------------------------
    # Load inputs
    # --------------------------------------------------
    articles_path = Path("data/processed/final_anomaly_results.csv")
    brands_path = Path("data/processed/article_brands.csv")
    output_path = Path("data/processed/brand_risk_scores.csv")

    articles = pd.read_csv(articles_path)
    brands = pd.read_csv(brands_path)

    print(f"✔ Articles loaded: {len(articles)}")
    print(f"✔ Brand links loaded: {len(brands)}")

    # --------------------------------------------------
    # Merge article → brand
    # --------------------------------------------------
    df = articles.merge(
        brands,
        on="article_id",
        how="inner"
    )

    print(f"✔ Merged rows: {len(df)}")

    # --------------------------------------------------
    # Convert anomaly flags to numeric
    # --------------------------------------------------
    df["linguistic_flag"] = df["is_anomaly"].map(
        {"Anomaly": 1, "Normal": 0}
    )

    df["location_flag"] = df["location_anomaly"].map(
        {"Anomaly": 1, "Review": 0.5, "Normal": 0}
    )

    df["temporal_flag"] = df["temporal_anomaly"].map(
        {"Anomaly": 1, "Normal": 0}
    )

    # --------------------------------------------------
    # Per-article risk score
    # --------------------------------------------------
    df["article_risk_score"] = (
        0.35 * df["linguistic_flag"] +
        0.25 * df["location_flag"] +
        0.15 * df["temporal_flag"] +
        0.25 * df["sentiment_negative"]
    )

    # --------------------------------------------------
    # Aggregate to brand level
    # --------------------------------------------------
    brand_risk = (
        df
        .groupby("organization")
        .agg(
            avg_article_risk=("article_risk_score", "mean"),
            article_count=("article_id", "nunique")
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Composite confidence-weighted risk
    # --------------------------------------------------
    brand_risk["brand_risk_score"] = (
        brand_risk["avg_article_risk"] *
        np.log1p(brand_risk["article_count"])
    )

    # --------------------------------------------------
    # Risk bands (dashboard friendly)
    # --------------------------------------------------
    def assign_risk_level(score):
        if score >= 1.2:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"

    brand_risk["risk_level"] = brand_risk["brand_risk_score"].apply(assign_risk_level)

    brand_risk = brand_risk.sort_values(
        "brand_risk_score",
        ascending=False
    )

    # --------------------------------------------------
    # Save output
    # --------------------------------------------------
    brand_risk.to_csv(output_path, index=False)

    print("✅ Composite brand risk completed")
    print("📁 Saved to:", output_path)
    print(brand_risk.head(10))


if __name__ == "__main__":
    main()

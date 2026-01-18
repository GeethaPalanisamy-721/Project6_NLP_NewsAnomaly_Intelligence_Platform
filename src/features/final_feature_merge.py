import pandas as pd

def main():
    print("🔗 Merging linguistic, location & temporal anomalies...")

    linguistic_df = pd.read_csv("data/processed/anomaly_scores.csv")
    location_df = pd.read_csv("data/processed/location_anomalies.csv")
    temporal_df = pd.read_csv("data/processed/news_with_temporal_anomaly.csv")

    # Start from linguistic anomaly file
    df = linguistic_df.copy()

    # Add other anomaly signals
    df["location_anomaly"] = location_df["location_anomaly"]
    df["temporal_anomaly"] = temporal_df["temporal_anomaly"]

    df.to_csv("data/processed/full_feature_set.csv", index=False)

    print("✅ full_feature_set.csv created successfully")
    print(df[["is_anomaly", "location_anomaly", "temporal_anomaly"]].head())

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os

def main():
    print("🚨 Running linguistic & semantic anomaly detection...")

    input_path = "data/processed/news_final_features.csv"
    output_path = "data/processed/anomaly_scores.csv"

    df = pd.read_csv(input_path)

    # 🔒 Collapse to article level
    df = (
        df.groupby("article_id", as_index=False)
        .agg(
            sentiment_label=("sentiment_label", "first"),
            topic_id=("topic_id", "first"),
            text_length=("text_length", "mean")
        )
    )

    # Encode sentiment
    le = LabelEncoder()
    df["sentiment_encoded"] = le.fit_transform(df["sentiment_label"])

    features = df[[
        "sentiment_encoded",
        "topic_id",
        "text_length"
    ]]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42
    )

    df["anomaly_score"] = model.fit_predict(features)
    df["is_anomaly"] = df["anomaly_score"].apply(
        lambda x: "Anomaly" if x == -1 else "Normal"
    )

    # Save minimal output
    os.makedirs("data/processed", exist_ok=True)
    df[["article_id", "is_anomaly", "anomaly_score"]].to_csv(
        output_path, index=False
    )

    print("✅ Linguistic anomaly detection completed")
    print(df["is_anomaly"].value_counts())
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    main()

#src/features/sentiment_analysis.py
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download once
nltk.download("vader_lexicon")


def analyze_sentiment(text):
    """
    Returns sentiment scores using VADER
    """
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series([0, 0, 0, "Neutral"])

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return pd.Series([
        scores["pos"],
        scores["neg"],
        scores["neu"],
        label
    ])


def main():
    print("😊 Running sentiment analysis...")

    df = pd.read_csv("data/processed/news_with_location.csv")

    if "clean_text" not in df.columns:
        raise ValueError("clean_text column missing")

    df[
        ["sentiment_positive", "sentiment_negative", "sentiment_neutral", "sentiment_label"]
    ] = df["clean_text"].apply(analyze_sentiment)

    df.to_csv("data/processed/news_with_sentiment.csv", index=False)

    print("✅ Sentiment analysis completed")
    print(df["sentiment_label"].value_counts())


if __name__ == "__main__":
    main()


# src/features/text_cleaning.py
"""
Text Cleaning & Schema Standardization
-------------------------------------
Purpose:
- Clean raw news article text
- Standardize column names early in the pipeline
- Create a clean, reliable base dataset for ALL downstream steps

This file defines the DATA CONTRACT for the entire project.
"""

# src/features/text_cleaning.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -------------------------------
# Download required NLTK data
# -------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# -------------------------------
# Main pipeline
# -------------------------------
def main():
    df = pd.read_csv(
        "data/raw/news_dataset.csv",
        encoding="latin-1",
        on_bad_lines="skip"
    )

    print(f"Total articles loaded: {len(df)}")

    # Create article_id ONCE
    df["article_id"] = range(len(df))

    # Clean ONLY the article body
    df["clean_text"] = df["Article"].apply(clean_text)

    # IMPORTANT: Do NOT drop Heading
    df.to_csv("data/processed/news_cleaned.csv", index=False)

    print("✅ Cleaned data saved to data/processed/news_cleaned.csv")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()

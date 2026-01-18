import pandas as pd
import spacy
from geotext import GeoText

# -----------------------------
# Load SpaCy NER model
# -----------------------------
nlp = spacy.load("en_core_web_sm")


# -----------------------------
# STEP 1: Location extraction
# -----------------------------
def extract_location(text):
    """
    Extract first meaningful location from text using:
    1. SpaCy NER (GPE / LOC)
    2. GeoText fallback
    """
    if not isinstance(text, str) or text.strip() == "":
        return "Unknown"

    # SpaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            return ent.text

    # GeoText fallback
    geo = GeoText(text)
    if geo.cities:
        return geo.cities[0]
    if geo.countries:
        return geo.countries[0]

    return "Unknown"


# -----------------------------
# STEP 2: Normalize location
# -----------------------------
def normalize_location(loc):
    """
    Normalize different variants of the same location
    into a canonical form.
    """
    if not isinstance(loc, str) or loc.strip() == "":
        return "Unknown"

    loc = loc.lower()

    mappings = {
        "us": "United States",
        "usa": "United States",
        "u.s.": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "uae": "United Arab Emirates",
    }

    for key, value in mappings.items():
        if key in loc:
            return value

    # Keep broad regions as-is (not anomalies by default)
    if loc in ["asia", "europe", "africa", "middle east"]:
        return loc.title()

    return loc.title()


# -----------------------------
# STEP 3: Smart anomaly logic
# -----------------------------
def detect_location_anomaly(claimed, content):
    """
    Business logic for location anomaly detection.
    """
    if claimed == "Unknown" and content == "Unknown":
        return "Normal"

    if claimed == "Unknown" and content != "Unknown":
        return "Normal"   # headlines often omit location

    if claimed != "Unknown" and content == "Unknown":
        return "Review"   # weak content evidence

    if claimed == content:
        return "Normal"

    return "Anomaly"


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("📍 Running correct location extraction (claim vs content)...")

    # Load cleaned data
    df = pd.read_csv("data/processed/news_cleaned.csv")

    required_cols = ["article_id", "Heading", "clean_text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} column missing. Run text_cleaning first.")

    # -----------------------------
    # Extract locations
    # -----------------------------
    df["claimed_location"] = df["Heading"].apply(extract_location)
    df["content_location"] = df["clean_text"].apply(extract_location)

    # Normalize
    df["claimed_location"] = df["claimed_location"].apply(normalize_location)
    df["content_location"] = df["content_location"].apply(normalize_location)

    # -----------------------------
    # Detect anomalies
    # -----------------------------
    df["location_anomaly"] = df.apply(
        lambda x: detect_location_anomaly(
            x["claimed_location"], x["content_location"]
        ),
        axis=1
    )

    # -----------------------------
    # Save output
    # -----------------------------
    output_path = "data/processed/news_with_location.csv"
    df.to_csv(output_path, index=False)

    print("✅ Location extraction completed")
    print(df["location_anomaly"].value_counts())


if __name__ == "__main__":
    main()

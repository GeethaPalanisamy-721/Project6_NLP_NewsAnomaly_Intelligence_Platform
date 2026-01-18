import pandas as pd
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

def extract_organizations(text):
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    return list(set(
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "ORG" and len(ent.text.strip()) > 2
    ))

def main():
    print("🏷️ Extracting organizations from RAW text (Heading + Article)...")

    base = Path("data/processed")
    df = pd.read_csv(base / "final_anomaly_results.csv")

    rows = []

    for _, row in df.iterrows():
        # ✅ RAW text — NOT clean_text
        raw_text = f"{row.get('Heading','')} {row.get('Article','')}"
        orgs = extract_organizations(raw_text)

        for org in orgs:
            rows.append({
                "article_id": row["article_id"],
                "organization": org
            })

    brand_df = pd.DataFrame(rows).drop_duplicates()

    brand_df.to_csv(base / "article_brands.csv", index=False)
    print(f"✅ Extracted {len(brand_df)} organization mentions")

if __name__ == "__main__":
    main()

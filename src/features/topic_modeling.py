import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic


def main():
    print("🧠 Running BERTopic modeling...")

    # Load data
    df = pd.read_csv("data/processed/news_with_sentiment.csv")

    if "clean_text" not in df.columns:
        raise ValueError("clean_text column missing")

    documents = df["clean_text"].astype(str).tolist()

    # Load embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=True,
        verbose=True
    )

    # Fit model
    topics, probs = topic_model.fit_transform(documents)

    # Assign topic info
    df["topic_id"] = topics
    df["topic_probability"] = probs.max(axis=1)
    df["topic_keywords"] = df["topic_id"].apply(
        lambda x: ", ".join([w for w, _ in topic_model.get_topic(x)][:5])
        if x != -1 else "Outlier"
    )

    # Save embeddings for anomaly models
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    df["embedding"] = embeddings.tolist()

    # Save output
    df.to_csv("data/processed/news_with_topics.csv", index=False)

    print("✅ BERTopic modeling completed")
    print(df["topic_id"].value_counts().head())


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import umap

def main():
    print("🔷 Running UMAP projection...")

    df = pd.read_csv("data/processed/news_with_topics.csv")

    if "embedding" not in df.columns:
        raise ValueError("Embedding column missing")

    # Convert embedding list-string → numpy array
    embeddings = np.vstack(df["embedding"].apply(eval).values)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    umap_embeddings = reducer.fit_transform(embeddings)

    umap_df = pd.DataFrame({
        "article_id": df["article_id"],
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1]
    })

    umap_df.to_csv("data/processed/umap_embeddings.csv", index=False)

    print("✅ UMAP projection saved → data/processed/umap_embeddings.csv")

if __name__ == "__main__":
    main()

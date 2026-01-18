#fianl with local filter, tab3 table
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="News Anomaly Intelligence Platform",
    layout="wide"
)

# ==================================================
# Load Data
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final_anomaly_results.csv")
    umap_df = pd.read_csv("data/processed/umap_embeddings.csv")
    topic_kw = pd.read_csv("data/processed/topic_keywords.csv")
    brand_df = pd.read_csv("data/processed/brand_risk_scores.csv")
    return df, umap_df, topic_kw, brand_df

df, umap_df, topic_kw, brand_df = load_data()

# Merge UMAP with article metadata
umap_df = umap_df.merge(
    df[["article_id", "Heading", "final_label", "content_location", "location_anomaly"]],
    on="article_id",
    how="inner"
)

# ==================================================
# SIDEBAR FILTERS
# ==================================================
st.sidebar.title("🔍 Global Filters")

location_options = ["All"] + sorted(
    df["location_clean"]
    .dropna()
    .loc[df["location_type"] != "UNKNOWN"]
    .unique()
)

selected_location = st.sidebar.selectbox(
    "Content Location",
    location_options
)

label_options = ["All"] + sorted(df["final_label"].unique())
selected_label = st.sidebar.selectbox("Risk Classification", label_options)

# 👉 NEW NewsType filter
news_type_options = ["All"] + sorted(df["NewsType"].dropna().unique())
selected_news_type = st.sidebar.selectbox("News Type", news_type_options)

# Apply filters
filtered_df = df.copy()
if selected_location != "All":
    filtered_df = filtered_df[filtered_df["content_location"] == selected_location]
if selected_label != "All":
    filtered_df = filtered_df[filtered_df["final_label"] == selected_label]
if selected_news_type != "All":
    filtered_df = filtered_df[filtered_df["NewsType"] == selected_news_type]


# ==================================================
# COLOR MAPS
# ==================================================
LABEL_COLORS = {
    "RED FLAG": "#c0392b",
    "REVIEW": "#f39c12",
    "NORMAL": "#2ecc71"
}

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Neutral": "#3498db",
    "Negative": "#c0392b"
}

RISK_COLORS = {
    "High": "#c0392b",
    "Medium": "#e67e22",
    "Low": "#2ecc71"
}

# ==================================================
# EXECUTIVE EXPLANATION — HOW THIS DASHBOARD WORKS
# ==================================================
st.markdown("## 🧭 How This Intelligence Platform Works")

logic_table = pd.DataFrame({
    "Dashboard Section": [
        "Disinformation Detection",
        "Hyperlocal Trend Monitoring",
        "Content Review Queue",
        "Brand Risk Intelligence"
    ],
    "Primary Signals Used": [
        "Linguistic, Location, Temporal anomalies + semantic clustering",
        "Topic dominance, sentiment trends, time-series patterns",
        "Total anomaly score, risk label, contextual signals",
        "Composite risk score, frequency of exposure, topic context"
    ],
    "What It Tells You": [
        "Early warning signals of coordinated or misleading narratives",
        "Which topics and sentiments dominate over time",
        "Which articles require immediate human verification",
        "Which brands face sustained reputational exposure"
    ]
})

st.dataframe(logic_table, use_container_width=True)

st.caption(
    "⚠️ This platform surfaces **risk signals and patterns**. "
    "Final verification decisions remain with human analysts."
)

st.divider()

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🛰️ Disinformation Detection",
    "📈 Hyperlocal Trend Monitoring",
    "🧾 Content Review Queue",
    "🏷️ Brand Risk Intelligence"
])
# ==================================================
# TAB 1 — Disinformation Detection
# ==================================================
with tab1:
    st.subheader("🛰️ Disinformation Detection Overview")

    # --------------------------------------------------
    # 1️⃣ KPI CARDS
    # --------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)

    total_articles = len(filtered_df)
    red_pct = (filtered_df["final_label"] == "RED FLAG").mean() * 100
    review_pct = (filtered_df["final_label"] == "REVIEW").mean() * 100
    location_anom_pct = (
        filtered_df["location_anomaly"] == "Anomaly"
    ).mean() * 100

    k1.metric("📰 Articles Monitored", f"{total_articles}")
    k2.metric("🚨 Red Flag (%)", f"{red_pct:.1f}%")
    k3.metric("⚠️ Review (%)", f"{review_pct:.1f}%")
    k4.metric("📍 Location Anomaly (%)", f"{location_anom_pct:.1f}%")

    st.divider()

    # --------------------------------------------------
    # 2️⃣ UMAP VISUALIZATION (Semantic Clusters)
    # --------------------------------------------------
    umap_filtered = umap_df.merge(
        filtered_df[["article_id", "location_anomaly"]],
        on="article_id",
        how="inner",
        suffixes=("", "_drop")
    )
    # remove duplicate column if created
    umap_filtered = umap_filtered.drop(
    columns=[c for c in umap_filtered.columns if c.endswith("_drop")],
    errors="ignore"
    )   
    left, right = st.columns([4, 2])

    with left:
        fig_umap = px.scatter(
            umap_filtered,
            x="x",
            y="y",
            color="final_label",
            color_discrete_map=LABEL_COLORS,
            hover_data=[
                "Heading",
                "content_location",
                "location_anomaly"
            ],
            title="Semantic Clusters Highlighting Location-Based Anomalies"
        )

        fig_umap.update_traces(
            marker=dict(size=6, opacity=0.75)
        )

        fig_umap.update_layout(
            legend_title_text="Risk Classification"
        )

        st.plotly_chart(fig_umap, use_container_width=True)

    # --------------------------------------------------
    # 3️⃣ INTERPRETATION PANEL
    # --------------------------------------------------
    with right:
        st.markdown("### Interpretation")

        st.markdown("""
**Each dot represents one news article**

• Articles close together share similar meaning  
• **Red Flag clusters** indicate coordinated or abnormal narratives  
• **Review clusters** show emerging or ambiguous patterns  
• Location mismatches inside clusters may signal disinformation  

**Analyst focus:**  
Dense clusters dominated by RED / REVIEW labels that contradict
their reported locations.
        """)

    st.divider()

    # --------------------------------------------------
    # 4️⃣ RISK DISTRIBUTION SUMMARY
    # --------------------------------------------------
    st.markdown("### 🔍 Risk Concentration Summary")

    risk_distribution = (
        umap_filtered
        .groupby("final_label")
        .size()
        .reset_index(name="article_count")
    )

    fig_dist = px.bar(
        risk_distribution,
        x="final_label",
        y="article_count",
        color="final_label",
        color_discrete_map=LABEL_COLORS,
        title="Distribution of Articles by Risk Classification"
    )

    fig_dist.update_layout(
        xaxis_title="Risk Label",
        yaxis_title="Number of Articles"
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    st.info(
        "🔎 **Insight:** High-risk clusters with semantic similarity and "
        "location inconsistencies often indicate recycled or manipulated narratives."
    )

# ==================================================
# TAB 2 — Hyperlocal Trend Monitoring
# ==================================================
with tab2:
    st.subheader("Sentiment & Topic Evolution Over Time")

    # Sentiment Trend
    trend_df = (
        filtered_df
        .groupby(["year", "sentiment_label"])
        .size()
        .reset_index(name="article_count")
    )

    fig_trend = px.line(
        trend_df,
        x="year",
        y="article_count",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        markers=True,
        title="Sentiment Trends Over Time"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Emerging Topics (Top 10)
    topic_counts = (
        df[df["topic_keywords"] != "Outlier"]
        .groupby("topic_keywords")
        .size()
        .reset_index(name="article_count")
        .sort_values("article_count", ascending=False)
        .head(10)
    )

    fig_topics = px.bar(
        topic_counts,
        x="article_count",
        y="topic_keywords",
        orientation="h",
        title="Top 10 Emerging Topics"
    )
    fig_topics.update_layout(
        xaxis_title="Number of Articles",
        yaxis_title="Topic Keywords",
        title={
            "text": "Top 10 Emerging Topics<br><sup>Outlier topics excluded for clarity</sup>",
            "x": 0.5
        }
    )
    st.plotly_chart(fig_topics, use_container_width=True)

# ==================================================
# TAB 3 — Content Review Queue
# ==================================================
with tab3:
    st.subheader("Articles Requiring Human Review")

    review_df = filtered_df[
        filtered_df["final_label"].isin(["RED FLAG", "REVIEW"])
    ][[
    "Heading",
    "content_location",
    "location_anomaly",
    "sentiment_label",
    "is_anomaly",
    "temporal_anomaly",
    "final_label",
    "total_anomaly_score"
    ]].sort_values("total_anomaly_score", ascending=False)

# Rename columns only for dashboard display
    review_df = review_df.rename(columns={
    "is_anomaly": "Linguistic Anomaly",
    "temporal_anomaly": "Temporal Anomaly",
    "final_label": "Final Risk Label",
    "total_anomaly_score": "Total Anomaly Score"
    })

    

    st.dataframe(review_df, use_container_width=True)

    st.caption(
        "These articles triggered one or more anomaly signals "
        "and should be manually verified by analysts."
    )

# ==================================================
# TAB 4 — Brand Risk Intelligence
# ==================================================
with tab4:
    st.subheader("Brand Risk Assessment & Explainability")

    left, right = st.columns([3, 2])

    # -----------------------------
    # LEFT: Risk Table
    # -----------------------------
    with left:
        st.markdown("### 📊 Brand Risk Summary")
        st.dataframe(
            brand_df.sort_values("brand_risk_score", ascending=False),
            use_container_width=True
        )

    # -----------------------------
    # RIGHT: Explainability
    # -----------------------------
    with right:
        st.markdown("### 🧠 Risk Scoring Logic")
        st.markdown("""
**Article Risk Score**
- 0.35 × Linguistic Anomaly  
- 0.25 × Location Anomaly  
- 0.15 × Temporal Anomaly  
- 0.25 × Negative Sentiment  

**Brand Risk Score**  
Brand Risk = Avg Article Risk × log(1 + Article Count)

**Why this works**  
- Single article ≠ reputation risk  
- Repeated negative coverage ↑ confidence  
- Log scaling prevents volume bias
        """)

    st.divider()

    # -----------------------------
    # Brand Risk Bar Chart
    # -----------------------------
    fig_brand = px.bar(
        brand_df.sort_values("brand_risk_score", ascending=False).head(15),
        x="organization",
        y="brand_risk_score",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        hover_data=["avg_article_risk", "article_count"],
        title="Top Brands by Composite Risk Score"
    )
    fig_brand.update_layout(
        xaxis_title="Organization",
        yaxis_title="Brand Risk Score",
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig_brand, use_container_width=True)

    # -----------------------------
    # Risk Context: Common Topics
    # -----------------------------
    st.divider()
    st.markdown("### 🧩 Risk Context: Common Topics")

    articles = pd.read_csv("data/processed/final_anomaly_results.csv")

    topic_context = (
        articles
        .explode("topic_keywords")
        .groupby("topic_keywords")
        .size()
        .reset_index(name="article_count")
        .sort_values("article_count", ascending=False)
        .head(10)
    )

    st.dataframe(topic_context, use_container_width=True)

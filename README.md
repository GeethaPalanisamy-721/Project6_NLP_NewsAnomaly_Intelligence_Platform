# 📰 News Anomaly Intelligence Platform

## Overview
The **News Anomaly Intelligence Platform** is designed to detect, visualize, and aggregate signals of potential disinformation or reputational risk in online news coverage.  
It helps analysts and stakeholders move from **article-level anomalies** to **brand-level risk intelligence**, providing both transparency and actionable insights.

## 🎯 Why This Matters
- **For analysts:** Quickly identify suspicious articles that require human review.
- **For clients/stakeholders:** Understand which brands face reputational exposure and why.
- **For decision-makers:** Gain early warning signals of coordinated or misleading narratives before they escalate.

## 🧩 Key Features
- **Disinformation Detection (Tab 1):**  
  UMAP semantic clustering highlights unusual articles and location mismatches.
- **Hyperlocal Trend Monitoring (Tab 2):**  
  Sentiment trends and emerging topics tracked over time.
- **Content Review Queue (Tab 3):**  
  Rule-based anomaly labeling (Normal, Review, Red Flag) for analyst triage.
- **Brand Risk Intelligence (Tab 4):**  
  Weighted risk scoring aggregated to brand level, with explainability panels.

## ⚙️ Technical Workflow
1. **Data Handling:**  
   - News articles ingested and cleaned (text normalization, metadata extraction).
2. **Feature Engineering:**  
   - Linguistic features (length, encoding artifacts, unusual phrasing).  
   - Location features (NER-based mismatch between headline and body).  
   - Temporal features (spikes in article volume).  
   - Sentiment features (VADER polarity scores).
3. **Anomaly Detection:**  
   - Isolation Forest for linguistic anomalies.  
   - Rule-based checks for location mismatches.  
   - Z-score detection for temporal spikes.  
   - Sentiment classification (Positive, Neutral, Negative).
4. **Aggregation:**  
   - Tab 3: Simple anomaly count → article labels.  
   - Tab 4: Weighted risk formula → brand risk scores.
5. **Visualization:**  
   - UMAP clusters for semantic similarity.  
   - Line charts for sentiment trends.  
   - Bar charts for topic dominance and brand risk.
6. **Deployment:**  
   - Streamlit dashboard with interactive filters.  
   - AWS-ready structure for client delivery.


## 📊 Risk Scoring Logic
**Article Risk Score**  :  0.35 × Linguistic Anomaly , 0.25 × Location Anomaly , 0.15 × Temporal Anomaly , 0.25 × Negative Sentiment
**Brand Risk Score**  :  = Avg Article Risk × log(1 + Article Count)
- **Balanced weighting:** Linguistic anomalies carry the most weight, temporal spikes the least.  
- **Explainable:** Each weight reflects the relative importance of signals in disinformation detection.  
- **Business relevance:** Converts technical anomalies into reputational risk categories (Low, Medium, High).

## 🚀 How It Helps Clients
- **Transparency:** Clear logic for why an article or brand is flagged.  
- **Efficiency:** Analysts focus on Red Flags and Reviews instead of all articles.  
- **Strategic insight:** Stakeholders see which brands are most exposed to reputational risk.  
- **Early warning:** Detects suspicious narratives before they spread widely.

## 🛠️ Tech Stack
- **Python** (data processing, anomaly detection)  
- **Pandas / NumPy** (feature engineering, aggregation)  
- **Sentence-BERT / BERTopic** (semantic embeddings, topic modeling)  
- **UMAP + HDBSCAN** (clustering and visualization)  
- **Plotly / Streamlit** (interactive dashboard)  
- **AWS / Docker** (deployment-ready architecture)

## 📈 Example Outputs
- **UMAP scatterplot:** Articles clustered by meaning, anomalies highlighted.  
- **Sentiment trend chart:** Positive/Negative/Neutral coverage over time.  
- **Review queue table:** Articles flagged for analyst verification.  
- **Brand risk bar chart:** Top organizations ranked by composite risk score.

## ⚠️ Disclaimer
This platform surfaces **risk signals and patterns**.  
Final verification decisions remain with **human analysts**.

**Where Users Can Get Help?** For specific questions, open a new issue with detailed descriptions. You can reach out via email at: geethabalan96@gmail.com, Kindly check NLP News Intelligence Anomaly radar.pptx for reference.

Who Maintains and Contributes to the Project? Author: Geetha Palanisamy
Project Type: Academic Deep Learning Project

**Guidance:** Developed through practical experimentation in anomaly detection, model evaluation, and dashboard deployment using Python, TensorFlow, and Streamlit. The system was iteratively refined using real-world news data, focusing on explainability, semantic clustering (UMAP), and risk scoring logic. Evaluation metrics (Precision, Recall, ROC_AUC) were used to validate anomaly detection performance, while stakeholder feedback shaped the dashboard’s usability and interpretability.
Contributions, suggestions, and pull requests are welcome — please open an issue before making major changes



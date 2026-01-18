import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
import os

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = "data/processed/final_anomaly_output.csv"
OUTPUT_DIR = "data/evaluation"
OUTPUT_FILE = f"{OUTPUT_DIR}/evaluation_summary.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Use proxy ground truth
df_eval = df[df["final_label"].isin(["RED FLAG", "NORMAL"])].copy()

df_eval["y_true"] = df_eval["final_label"].map({
    "RED FLAG": 1,
    "NORMAL": 0
})

# Composite anomaly score
df_eval["anomaly_score"] = (
    (df_eval["is_anomaly"] == "Anomaly").astype(int) +
    (df_eval["location_anomaly"] == "Anomaly").astype(int) +
    (df_eval["temporal_anomaly"] == "Anomaly").astype(int)
) / 3

# Binary prediction using threshold
THRESHOLD = 0.7
df_eval["y_pred"] = (df_eval["anomaly_score"] >= THRESHOLD).astype(int)

# ----------------------------
# METRICS
# ----------------------------
roc_auc = roc_auc_score(df_eval["y_true"], df_eval["anomaly_score"])
pr_auc = average_precision_score(df_eval["y_true"], df_eval["anomaly_score"])
precision = precision_score(df_eval["y_true"], df_eval["y_pred"])
recall = recall_score(df_eval["y_true"], df_eval["y_pred"])
f1 = f1_score(df_eval["y_true"], df_eval["y_pred"])

# ----------------------------
# RECALL@K
# ----------------------------
def recall_at_k(df, k):
    top_k = df.sort_values("anomaly_score", ascending=False).head(k)
    return top_k["y_true"].sum() / df["y_true"].sum()

recall_50 = recall_at_k(df_eval, 50)
recall_100 = recall_at_k(df_eval, 100)
recall_200 = recall_at_k(df_eval, 200)

# ----------------------------
# SAVE RESULTS AS CSV
# ----------------------------
results = pd.DataFrame({
    "metric": [
        "ROC_AUC",
        "PR_AUC",
        "Precision",
        "Recall",
        "F1_score",
        "Recall@50",
        "Recall@100",
        "Recall@200"
    ],
    "value": [
        roc_auc,
        pr_auc,
        precision,
        recall,
        f1,
        recall_50,
        recall_100,
        recall_200
    ]
})

results.to_csv(OUTPUT_FILE, index=False)

# ----------------------------
# PRINT SUMMARY
# ----------------------------
print("📊 Evaluation Metrics Summary")
print(results)
print(f"\n✅ Evaluation results saved to: {OUTPUT_FILE}")

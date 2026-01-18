import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = "data/processed/final_anomaly_output.csv"
OUTPUT_DIR = "data/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Proxy labels
df_eval = df[df["final_label"].isin(["RED FLAG", "NORMAL"])].copy()
df_eval["y_true"] = df_eval["final_label"].map({
    "RED FLAG": 1,
    "NORMAL": 0
})

# Anomaly score (same as evaluation)
df_eval["anomaly_score"] = (
    (df_eval["is_anomaly"] == "Anomaly").astype(int) +
    (df_eval["location_anomaly"] == "Anomaly").astype(int) +
    (df_eval["temporal_anomaly"] == "Anomaly").astype(int)
) / 3

# ----------------------------
# ROC CURVE
# ----------------------------
fpr, tpr, _ = roc_curve(df_eval["y_true"], df_eval["anomaly_score"])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Anomaly Detection")
plt.legend()
plt.grid(True)

plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
plt.close()

# ----------------------------
# PRECISION-RECALL CURVE
# ----------------------------
precision, recall, _ = precision_recall_curve(
    df_eval["y_true"], df_eval["anomaly_score"]
)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Anomaly Detection")
plt.legend()
plt.grid(True)

plt.savefig(f"{OUTPUT_DIR}/pr_curve.png")
plt.close()

print("📈 ROC and PR curves generated and saved successfully!")

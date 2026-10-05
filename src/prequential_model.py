import pandas as pd
import numpy as np

from river import compose
from river import preprocessing
from river import linear_model
from river import metrics


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/processed/preprocessed_fraud_data.csv"

# Number of transactions to process.
# Start small for testing.
MAX_TRANSACTIONS = 5000


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 4 - PREQUENTIAL FRAUD DETECTION")
print("=" * 60)

print("\nLoading preprocessed data...")

data = pd.read_csv(DATA_PATH)

# Make sure chronological order is maintained
data = data.sort_values(
    "TransactionDT"
).reset_index(drop=True)

print("Dataset shape:", data.shape)


# ============================================================
# LIMIT DATA FOR INITIAL TEST
# ============================================================

if MAX_TRANSACTIONS is not None:
    data = data.iloc[:MAX_TRANSACTIONS].copy()

print("Transactions used:", len(data))


# ============================================================
# SELECT FEATURES
# ============================================================

feature_columns = [
    "TransactionDT",
    "TransactionAmt",
    "TransactionAmt_log",

    "TransactionHour",
    "TransactionDay",
    "TransactionWeek",

    "ProductCD",

    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    "addr1",
    "addr2",

    "dist1",
    "dist2",

    "P_emaildomain",
    "R_emaildomain",

    "C1",
    "C2",
    "C5",
    "C6",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",

    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D10",
    "D15",

    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",

    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
    "id_13",
    "id_17",
    "id_19",
    "id_20",

    "id_31",
    "id_36",
    "id_37",
    "id_38",

    "DeviceType",
    "DeviceInfo"
]


# Keep only columns that actually exist
feature_columns = [
    col for col in feature_columns
    if col in data.columns
]

print("\nNumber of selected features:", len(feature_columns))


# ============================================================
# PREPARE DATA
# ============================================================

X = data[feature_columns]
y = data["isFraud"]


# ============================================================
# CONVERT DATA FOR RIVER
# ============================================================

def convert_value(value):

    if pd.isna(value):
        return 0.0

    if isinstance(value, str):
        return value

    return float(value)


def row_to_dict(row):

    return {
        column: convert_value(row[column])
        for column in feature_columns
    }


# ============================================================
# MODEL
# ============================================================

model = compose.Pipeline(
    preprocessing.OneHotEncoder(),
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)


# ============================================================
# METRICS
# ============================================================

accuracy = metrics.Accuracy()
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1()
roc_auc = metrics.ROCAUC()


# ============================================================
# PREQUENTIAL EVALUATION
# ============================================================

print("\nStarting prequential evaluation...")

fraud_count = 0

for i in range(len(data)):

    # ----------------------------------------
    # CURRENT TRANSACTION
    # ----------------------------------------

    x = row_to_dict(X.iloc[i])

    target = int(y.iloc[i])

    if target == 1:
        fraud_count += 1


    # ----------------------------------------
    # TEST FIRST
    # ----------------------------------------

    prediction = model.predict_one(x)

    probability = model.predict_proba_one(x)

    if prediction is not None:

        accuracy.update(target, prediction)
        precision.update(target, prediction)
        recall.update(target, prediction)
        f1.update(target, prediction)

        fraud_probability = probability.get(1, 0.0)

        roc_auc.update(
            target,
            fraud_probability
        )


    # ----------------------------------------
    # TRAIN AFTER TEST
    # ----------------------------------------

    model.learn_one(x, target)


    # ----------------------------------------
    # PROGRESS
    # ----------------------------------------

    if (i + 1) % 500 == 0:

        print(
            f"Processed: {i + 1:,} | "
            f"Fraud: {fraud_count:,} | "
            f"Accuracy: {accuracy.get():.4f} | "
            f"Precision: {precision.get():.4f} | "
            f"Recall: {recall.get():.4f} | "
            f"F1: {f1.get():.4f}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PREQUENTIAL EVALUATION COMPLETE")
print("=" * 60)

print(f"\nTransactions processed: {len(data):,}")
print(f"Fraud transactions:     {fraud_count:,}")

print(f"\nAccuracy:  {accuracy.get():.4f}")
print(f"Precision: {precision.get():.4f}")
print(f"Recall:    {recall.get():.4f}")
print(f"F1 Score:  {f1.get():.4f}")
print(f"ROC-AUC:   {roc_auc.get():.4f}")
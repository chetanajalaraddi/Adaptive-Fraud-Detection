import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

TRANSACTION_PATH = "data/raw/train_transaction.csv"
IDENTITY_PATH = "data/raw/train_identity.csv"

OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 3 - IEEE-CIS DATA PREPROCESSING")
print("=" * 70)

print("\n[1/8] Loading transaction data...")

transactions = pd.read_csv(TRANSACTION_PATH)

print("Transaction shape:", transactions.shape)

print("\n[2/8] Loading identity data...")

identity = pd.read_csv(IDENTITY_PATH)

print("Identity shape:", identity.shape)


# ============================================================
# MERGE
# ============================================================

print("\n[3/8] Merging transaction + identity data...")

data = transactions.merge(
    identity,
    on="TransactionID",
    how="left"
)

print("Merged shape:", data.shape)


# ============================================================
# TEMPORAL ORDER
# ============================================================

print("\n[4/8] Sorting chronologically...")

data = data.sort_values(
    by=["TransactionDT", "TransactionID"]
).reset_index(drop=True)


# ============================================================
# TARGET CHECK
# ============================================================

print("\nFraud distribution:")

fraud_counts = data["isFraud"].value_counts()

print(fraud_counts)

print("\nFraud percentage:")

print(
    data["isFraud"]
    .value_counts(normalize=True)
    .mul(100)
)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

print("\n[5/8] Creating temporal features...")

data["TransactionHour"] = (
    (data["TransactionDT"] // 3600) % 24
).astype("int16")

data["TransactionDay"] = (
    data["TransactionDT"] // (3600 * 24)
).astype("int32")

data["TransactionWeek"] = (
    data["TransactionDay"] // 7
).astype("int16")

data["TransactionDayOfWeek"] = (
    data["TransactionDay"] % 7
).astype("int8")


# Circular time representation
data["TransactionHourSin"] = np.sin(
    2 * np.pi * data["TransactionHour"] / 24
)

data["TransactionHourCos"] = np.cos(
    2 * np.pi * data["TransactionHour"] / 24
)


# ============================================================
# TRANSACTION AMOUNT FEATURES
# ============================================================

print("\n[6/8] Creating transaction amount features...")

data["TransactionAmt_log"] = np.log1p(
    data["TransactionAmt"].clip(lower=0)
)


# ============================================================
# BALANCE / TRANSACTION FEATURES
# ============================================================

# These features are available only where the corresponding
# columns exist.

if "C1" in data.columns:
    data["C1_log"] = np.log1p(
        data["C1"].clip(lower=0)
    )

if "C2" in data.columns:
    data["C2_log"] = np.log1p(
        data["C2"].clip(lower=0)
    )


# ============================================================
# MISSING VALUE INDICATORS
# ============================================================

print("\n[7/8] Creating missing-value indicators...")

important_columns = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_01",
    "id_02",
    "id_31",
    "id_36",
]

for column in important_columns:

    if column in data.columns:

        data[f"{column}_missing"] = (
            data[column]
            .isna()
            .astype("int8")
        )


# ============================================================
# PRESERVE TRANSACTION ID
# ============================================================

# IMPORTANT:
# TransactionID is retained for traceability and evaluation.
# It will NOT be used as an ML feature.

# Do NOT drop TransactionID here.


# ============================================================
# SAVE
# ============================================================

print("\n[8/8] Saving processed dataset...")

output_path = os.path.join(
    OUTPUT_DIR,
    "ieee_preprocessed.csv"
)

data.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nFinal shape:")
print(data.shape)

print("\nTarget distribution:")
print(data["isFraud"].value_counts())

print("\nFraud percentage:")

print(
    data["isFraud"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nTransaction time range:")

print(
    "Start:",
    data["TransactionDT"].min()
)

print(
    "End:",
    data["TransactionDT"].max()
)

print("\nTransactionID preserved:", "TransactionID" in data.columns)

print("\nDone.")
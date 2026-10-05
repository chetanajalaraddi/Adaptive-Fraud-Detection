import os
import time
import numpy as np
import pandas as pd


# ============================================================
# STEP 14 - CREDIT CARD FRAUD DATASET PREPROCESSING
# ============================================================

print("=" * 70)
print("STEP 14 - CREDIT CARD FRAUD DATASET PREPROCESSING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/raw/creditcard.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "creditcard_features.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# START TIMER
# ============================================================

start_time = time.time()


# ============================================================
# [1/7] LOADING DATASET
# ============================================================

print("\n[1/7] Loading Credit Card Fraud dataset...")

data = pd.read_csv(
    INPUT_PATH
)

print(
    "Original shape:",
    data.shape
)


# ============================================================
# [2/7] CHECKING COLUMNS
# ============================================================

print("\n[2/7] Checking dataset structure...")

required_columns = [
    "Time",
    "Amount",
    "Class"
]

for column in required_columns:

    if column not in data.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


print(
    "Required columns:",
    required_columns
)

print(
    "Total columns:",
    len(data.columns)
)


# ============================================================
# [3/7] CHRONOLOGICAL SORTING
# ============================================================

print(
    "\n[3/7] Sorting transactions chronologically..."
)

data = data.sort_values(
    by=[
        "Time"
    ],
    kind="stable"
).reset_index(
    drop=True
)


# ============================================================
# CREATE TRANSACTION ID
# ============================================================

data["TransactionID"] = (
    np.arange(
        1,
        len(data) + 1
    )
)


# ============================================================
# RENAME TIMESTAMP
# ============================================================

data["TransactionDT"] = (
    data["Time"]
)


# ============================================================
# RENAME TARGET
# ============================================================

data["isFraud"] = (
    data["Class"]
)


# ============================================================
# [4/7] SELECT FEATURES
# ============================================================

print(
    "\n[4/7] Selecting model features..."
)


feature_columns = [
    column
    for column in data.columns
    if column.startswith("V")
]


feature_columns.append(
    "Amount"
)


print(
    "PCA features:",
    len(
        [
            c
            for c in feature_columns
            if c.startswith("V")
        ]
    )
)

print(
    "Total model features:",
    len(feature_columns)
)


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

processed_columns = [
    "TransactionID",
    "TransactionDT"
]

processed_columns.extend(
    feature_columns
)

processed_columns.append(
    "isFraud"
)


data = data[
    processed_columns
].copy()


# ============================================================
# [5/7] DATA CLEANING
# ============================================================

print(
    "\n[5/7] Cleaning numerical data..."
)


numeric_columns = [
    column
    for column in data.columns
    if column not in [
        "TransactionID",
        "isFraud"
    ]
]


# Convert numerical columns safely

for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# Replace infinity

data = data.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


# Count missing values

missing_before = int(
    data.isna()
    .sum()
    .sum()
)


print(
    "Missing values before cleaning:",
    missing_before
)


# Fill numerical missing values

data[numeric_columns] = (
    data[numeric_columns]
    .fillna(0.0)
)


missing_after = int(
    data.isna()
    .sum()
    .sum()
)


print(
    "Missing values after cleaning:",
    missing_after
)


# ============================================================
# TARGET VALIDATION
# ============================================================

data["isFraud"] = (
    pd.to_numeric(
        data["isFraud"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# [6/7] DATASET STATISTICS
# ============================================================

print(
    "\n[6/7] Calculating dataset statistics..."
)


total_transactions = len(
    data
)


fraud_count = int(
    data["isFraud"]
    .sum()
)


normal_count = (
    total_transactions
    -
    fraud_count
)


fraud_percentage = (
    fraud_count
    /
    total_transactions
    *
    100
)


print(
    "\nDataset Summary"
)

print(
    "-" * 50
)

print(
    "Transactions       :",
    f"{total_transactions:,}"
)

print(
    "Normal transactions:",
    f"{normal_count:,}"
)

print(
    "Fraud transactions :",
    f"{fraud_count:,}"
)

print(
    "Fraud percentage   :",
    f"{fraud_percentage:.4f}%"
)


# ============================================================
# TIME RANGE
# ============================================================

print(
    "\nChronological information"
)

print(
    "-" * 50
)

print(
    "First Time:",
    data["TransactionDT"].iloc[0]
)

print(
    "Last Time :",
    data["TransactionDT"].iloc[-1]
)


# ============================================================
# CHECK CHRONOLOGICAL ORDER
# ============================================================

is_chronological = (
    data["TransactionDT"]
    .is_monotonic_increasing
)


print(
    "Chronological order:",
    "OK" if is_chronological else "FAILED"
)


if not is_chronological:

    raise ValueError(
        "Dataset is not chronologically ordered."
    )


# ============================================================
# CHECK TRANSACTION IDs
# ============================================================

transaction_ids_unique = (
    data["TransactionID"]
    .is_unique
)


print(
    "Transaction IDs unique:",
    "OK"
    if transaction_ids_unique
    else "FAILED"
)


if not transaction_ids_unique:

    raise ValueError(
        "TransactionID values are not unique."
    )


# ============================================================
# CHECK TARGET VALUES
# ============================================================

target_values = sorted(
    data["isFraud"]
    .unique()
)


print(
    "Target values:",
    target_values
)


if not set(
    target_values
).issubset(
    {0, 1}
):

    raise ValueError(
        "Target contains values other than 0 and 1."
    )


# ============================================================
# [7/7] SAVING PROCESSED DATASET
# ============================================================

print(
    "\n[7/7] Saving processed dataset..."
)


data.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# VERIFY SAVED FILE
# ============================================================

saved_size = os.path.getsize(
    OUTPUT_PATH
)


print(
    "\nSaved successfully:"
)

print(
    OUTPUT_PATH
)

print(
    "Output shape:",
    data.shape
)

print(
    "Output size:",
    f"{saved_size / (1024 * 1024):.2f} MB"
)


# ============================================================
# FINAL COLUMN LIST
# ============================================================

print(
    "\nFinal columns:"
)

print(
    data.columns.tolist()
)


# ============================================================
# SAMPLE
# ============================================================

print(
    "\nFirst 5 processed transactions:"
)

print(
    data.head().to_string(
        index=False
    )
)


# ============================================================
# PROCESSING TIME
# ============================================================

elapsed_time = (
    time.time()
    -
    start_time
)


print(
    "\n" + "=" * 70
)

print(
    "STEP 14 COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Processing time: {elapsed_time:.2f} seconds"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_PATH
)

print(
    "\nDone."
)
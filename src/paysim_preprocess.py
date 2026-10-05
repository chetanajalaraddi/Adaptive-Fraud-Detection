import os
import time
import numpy as np
import pandas as pd


# ============================================================
# STEP 16 - PAYSIM FRAUD DATASET PREPROCESSING
# ============================================================

print("=" * 70)
print("STEP 16 - PAYSIM FRAUD DATASET PREPROCESSING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
    "data/raw/"
    "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = "data/processed"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "paysim_features.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


start_time = time.time()


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n[1/8] Loading PaySim dataset..."
)

data = pd.read_csv(
    INPUT_PATH
)

print(
    "Original shape:",
    data.shape
)


# ============================================================
# CHECK STRUCTURE
# ============================================================

print(
    "\n[2/8] Checking dataset structure..."
)

required_columns = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
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
# SORT CHRONOLOGICALLY
# ============================================================

print(
    "\n[3/8] Sorting transactions chronologically..."
)

data = data.sort_values(
    by=[
        "step"
    ]
).reset_index(
    drop=True
)

print(
    "Chronological order: OK"
)


# ============================================================
# TRANSACTION ID
# ============================================================

print(
    "\n[4/8] Creating transaction identifiers..."
)

data["TransactionID"] = (
    np.arange(
        1,
        len(data) + 1
    )
)

data["TransactionDT"] = (
    data["step"].astype(float)
)


# ============================================================
# TRANSACTION TYPE ENCODING
# ============================================================

print(
    "\n[5/8] Encoding transaction types..."
)

type_mapping = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4
}

data["transaction_type"] = (
    data["type"]
    .map(type_mapping)
    .fillna(-1)
    .astype(float)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print(
    "\n[6/8] Creating model features..."
)

data["origin_balance_error"] = (
    data["oldbalanceOrg"]
    -
    data["amount"]
    -
    data["newbalanceOrig"]
)

data["destination_balance_change"] = (
    data["newbalanceDest"]
    -
    data["oldbalanceDest"]
)

data["origin_balance_change"] = (
    data["oldbalanceOrg"]
    -
    data["newbalanceOrig"]
)

data["amount_to_origin_balance"] = (
    data["amount"]
    /
    (
        data["oldbalanceOrg"]
        +
        1.0
    )
)

data["amount_to_destination_balance"] = (
    data["amount"]
    /
    (
        data["oldbalanceDest"]
        +
        1.0
    )
)


# ============================================================
# SELECT MODEL FEATURES
# ============================================================

feature_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "transaction_type",
    "isFlaggedFraud",
    "origin_balance_error",
    "destination_balance_change",
    "origin_balance_change",
    "amount_to_origin_balance",
    "amount_to_destination_balance"
]

print(
    "Model features:",
    len(feature_columns)
)


# ============================================================
# CLEAN NUMERICAL DATA
# ============================================================

print(
    "\n[7/8] Cleaning numerical data..."
)

selected_columns = (
    [
        "TransactionID",
        "TransactionDT"
    ]
    +
    feature_columns
    +
    [
        "isFraud"
    ]
)

processed = data[
    selected_columns
].copy()


missing_before = int(
    processed.isna().sum().sum()
)

print(
    "Missing values before cleaning:",
    missing_before
)

for column in processed.columns:

    processed[column] = pd.to_numeric(
        processed[column],
        errors="coerce"
    )

processed = processed.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)

processed = processed.fillna(
    0.0
)

missing_after = int(
    processed.isna().sum().sum()
)

print(
    "Missing values after cleaning:",
    missing_after
)


# ============================================================
# DATASET STATISTICS
# ============================================================

fraud_count = int(
    processed["isFraud"].sum()
)

normal_count = (
    len(processed)
    -
    fraud_count
)

fraud_percentage = (
    fraud_count
    /
    len(processed)
    *
    100
)

print(
    "\nPaySim Dataset Summary"
)

print(
    "-" * 50
)

print(
    f"Transactions       : {len(processed):,}"
)

print(
    f"Normal transactions: {normal_count:,}"
)

print(
    f"Fraud transactions : {fraud_count:,}"
)

print(
    f"Fraud percentage   : {fraud_percentage:.4f}%"
)


# ============================================================
# CHRONOLOGICAL INFORMATION
# ============================================================

print(
    "\nChronological information"
)

print(
    "-" * 50
)

print(
    "First step:",
    processed["TransactionDT"].iloc[0]
)

print(
    "Last step :",
    processed["TransactionDT"].iloc[-1]
)

chronological = (
    processed["TransactionDT"]
    .is_monotonic_increasing
)

print(
    "Chronological order:",
    "OK"
    if chronological
    else "FAILED"
)

unique_ids = (
    processed["TransactionID"].is_unique
)

print(
    "Transaction IDs unique:",
    "OK"
    if unique_ids
    else "FAILED"
)

print(
    "Target values:",
    sorted(
        processed["isFraud"]
        .unique()
        .tolist()
    )
)


# ============================================================
# SAVE
# ============================================================

print(
    "\n[8/8] Saving processed dataset..."
)

processed.to_csv(
    OUTPUT_PATH,
    index=False
)

file_size_mb = (
    os.path.getsize(
        OUTPUT_PATH
    )
    /
    (
        1024 ** 2
    )
)

print(
    "\nSaved successfully:"
)

print(
    OUTPUT_PATH
)

print(
    f"Output shape: {processed.shape}"
)

print(
    f"Output size : {file_size_mb:.2f} MB"
)

print(
    "\nFinal columns:"
)

print(
    processed.columns.tolist()
)

print(
    "\nFirst 5 processed transactions:"
)

print(
    processed.head().to_string(
        index=False
    )
)


# ============================================================
# FINAL
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
    "STEP 16 COMPLETE"
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
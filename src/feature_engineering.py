import pandas as pd
import numpy as np


# ============================================================
# STEP 4 - STREAMING-SAFE FEATURE ENGINEERING
# ============================================================

print("=" * 70)
print("STEP 4 - STREAMING-SAFE FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_PATH = "data/processed/ieee_preprocessed.csv"
OUTPUT_PATH = "data/processed/ieee_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/6] Loading preprocessed dataset...")

data = pd.read_csv(INPUT_PATH)

print("Input shape:", data.shape)


# ============================================================
# BASIC FEATURES
# ============================================================

print("\n[2/6] Preparing model features...")


# ------------------------------------------------------------
# Amount features
# ------------------------------------------------------------

if "TransactionAmt" in data.columns:

    data["TransactionAmt_log"] = np.log1p(
        data["TransactionAmt"].clip(lower=0)
    )


# ------------------------------------------------------------
# Time features
# ------------------------------------------------------------

if "TransactionDT" in data.columns:

    data["TransactionHour"] = (
        (data["TransactionDT"] // 3600) % 24
    ).astype("int16")

    data["TransactionDay"] = (
        data["TransactionDT"] // (3600 * 24)
    ).astype("int32")

    data["TransactionDayOfWeek"] = (
        data["TransactionDay"] % 7
    ).astype("int8")

    data["TransactionHourSin"] = np.sin(
        2 * np.pi * data["TransactionHour"] / 24
    )

    data["TransactionHourCos"] = np.cos(
        2 * np.pi * data["TransactionHour"] / 24
    )


# ============================================================
# MISSING VALUE FEATURES
# ============================================================

print("\n[3/6] Creating missing-value indicators...")

important_columns = [
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
    "id_36"
]

for column in important_columns:

    if column in data.columns:

        data[f"{column}_missing"] = (
            data[column]
            .isna()
            .astype("int8")
        )


# ============================================================
# SAFE CATEGORICAL ENCODING
# ============================================================

print("\n[4/6] Encoding categorical variables...")


categorical_columns = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType"
]


for column in categorical_columns:

    if column in data.columns:

        # Convert categories to integer codes.
        # Missing values receive code -1.
        data[column] = (
            data[column]
            .astype("category")
            .cat.codes
            .astype("int32")
        )


# ============================================================
# SELECT MODEL FEATURES
# ============================================================

print("\n[5/6] Selecting model features...")


candidate_features = [

    # Transaction
    "TransactionAmt",
    "TransactionAmt_log",

    # Time
    "TransactionDT",
    "TransactionHour",
    "TransactionDay",
    "TransactionDayOfWeek",
    "TransactionHourSin",
    "TransactionHourCos",

    # Product
    "ProductCD",

    # Card
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    # Address
    "addr1",
    "addr2",

    # Email
    "P_emaildomain",
    "R_emaildomain",

    # Device
    "DeviceType",

    # Identity
    "id_01",
    "id_02",

    # Missing indicators
    "card1_missing",
    "card2_missing",
    "card3_missing",
    "card4_missing",
    "card5_missing",
    "card6_missing",
    "addr1_missing",
    "addr2_missing",
    "P_emaildomain_missing",
    "R_emaildomain_missing",
    "DeviceType_missing",
    "DeviceInfo_missing",
    "id_01_missing",
    "id_02_missing",
    "id_31_missing",
    "id_36_missing"
]


available_features = [
    column
    for column in candidate_features
    if column in data.columns
]


print(
    "Number of selected features:",
    len(available_features)
)


# ============================================================
# CREATE MODEL DATASET
# ============================================================

model_columns = [
    "TransactionID",
    "TransactionDT",
    "isFraud"
] + available_features


model_columns = list(
    dict.fromkeys(model_columns)
)


features = data[model_columns].copy()


# ============================================================
# HANDLE NUMERIC VALUES
# ============================================================

print("\nHandling missing numeric values...")

feature_columns = [
    column
    for column in available_features
    if column in features.columns
]


for column in feature_columns:

    if not pd.api.types.is_numeric_dtype(
        features[column]
    ):
        continue

    features[column] = (
        pd.to_numeric(
            features[column],
            errors="coerce"
        )
    )

    # Median calculated from the current processed dataset
    # is used only for preprocessing. Behavioral streaming
    # features will later be generated separately.
    median_value = features[column].median()

    if pd.isna(median_value):
        median_value = 0

    features[column] = (
        features[column]
        .fillna(median_value)
        .replace([np.inf, -np.inf], 0)
    )


# ============================================================
# FINAL SORT
# ============================================================

features = features.sort_values(
    by=["TransactionDT", "TransactionID"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

print("\n[6/6] Saving feature dataset...")

features.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("\nFinal shape:")
print(features.shape)

print("\nSelected features:")
for feature in available_features:
    print(" -", feature)

print("\nFraud distribution:")
print(features["isFraud"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nDone.")
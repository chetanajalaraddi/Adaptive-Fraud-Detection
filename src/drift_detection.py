import pandas as pd
import numpy as np

from river import compose
from river import preprocessing
from river import linear_model
from river import drift


# ============================================================
# STEP 6 - CONCEPT DRIFT DETECTION
# ============================================================

print("=" * 60)
print("STEP 6 - CONCEPT DRIFT DETECTION")
print("=" * 60)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/processed/preprocessed_fraud_data.csv"

# Start small for testing
MAX_TRANSACTIONS = 5000

# Rolling window for performance
WINDOW_SIZE = 500


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

data = pd.read_csv(DATA_PATH)

# IMPORTANT:
# Preserve chronological order
data = data.sort_values(
    "TransactionDT"
).reset_index(drop=True)

data = data.iloc[:MAX_TRANSACTIONS].copy()

print("Transactions:", len(data))


# ============================================================
# FEATURES
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

feature_columns = [
    column
    for column in feature_columns
    if column in data.columns
]


# ============================================================
# CONVERT VALUES
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
# ONLINE MODEL
# ============================================================

model = compose.Pipeline(
    preprocessing.OneHotEncoder(),
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)


# ============================================================
# DRIFT DETECTOR
# ============================================================

# ADWIN detects changes in the distribution of errors.
drift_detector = drift.ADWIN(
    delta=0.002
)


# ============================================================
# STORAGE
# ============================================================

errors = []

drift_points = []

rolling_f1_values = []

rolling_recall_values = []

rolling_precision_values = []


# ============================================================
# PREQUENTIAL EVALUATION
# ============================================================

print("\nStarting test-then-train drift detection...\n")


for i in range(len(data)):

    row = data.iloc[i]

    x = row_to_dict(row)

    y = int(row["isFraud"])


    # --------------------------------------------------------
    # TEST FIRST
    # --------------------------------------------------------

    prediction = model.predict_one(x)


    # Convert prediction to binary
    if prediction is None:
        prediction = 0


    prediction = int(prediction)


    # --------------------------------------------------------
    # CALCULATE ERROR
    # --------------------------------------------------------

    error = int(prediction != y)

    errors.append(error)


    # Feed error to ADWIN
    drift_detector.update(error)


    # --------------------------------------------------------
    # CHECK DRIFT
    # --------------------------------------------------------

    if drift_detector.drift_detected:

        drift_points.append(i + 1)

        print(
            f"\n⚠ DRIFT DETECTED at transaction "
            f"{i + 1:,}"
        )


    # --------------------------------------------------------
    # ROLLING METRICS
    # --------------------------------------------------------

    start = max(
        0,
        len(errors) - WINDOW_SIZE
    )

    recent_errors = errors[start:]

    recent_data = data.iloc[
        start:i + 1
    ]

    predictions = []

    actuals = []

    # We need predictions for rolling metrics.
    # For this simple first experiment,
    # use error-based accuracy and fraud recall later.

    # --------------------------------------------------------
    # TRAIN AFTER TEST
    # --------------------------------------------------------

    model.learn_one(x, y)


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (i + 1) % 500 == 0:

        accuracy = 1 - np.mean(
            recent_errors
        )

        print(
            f"Processed: {i + 1:,}/"
            f"{len(data):,} | "
            f"Rolling Accuracy: {accuracy:.4f}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DRIFT DETECTION RESULTS")
print("=" * 60)


overall_accuracy = 1 - np.mean(errors)


print(
    f"\nOverall Accuracy: "
    f"{overall_accuracy:.4f}"
)


print(
    f"\nNumber of drift events: "
    f"{len(drift_points)}"
)


if len(drift_points) > 0:

    print("\nDrift points:")

    for point in drift_points:

        print(
            f"  Transaction {point:,}"
        )

else:

    print(
        "\nNo drift detected in this "
        "5,000-transaction test."
    )


# ============================================================
# SAVE DRIFT POINTS
# ============================================================

drift_output = pd.DataFrame({
    "drift_transaction": drift_points
})


drift_output.to_csv(
    "data/processed/drift_points.csv",
    index=False
)


print(
    "\nDrift points saved to:"
)

print(
    "data/processed/drift_points.csv"
)


print("\n" + "=" * 60)
print("STEP 6 COMPLETE")
print("=" * 60)
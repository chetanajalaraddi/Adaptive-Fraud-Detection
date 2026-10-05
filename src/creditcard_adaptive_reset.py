import os
import time
import numpy as np
import pandas as pd

from river import linear_model
from river import preprocessing
from river import drift


# ============================================================
# STEP 15 - CREDIT CARD ADWIN + RESET ADAPTIVE MODEL
# ============================================================

print("=" * 70)
print("STEP 15 - CREDIT CARD ADWIN + RESET ADAPTIVE MODEL")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/processed/creditcard_features.csv"

RESULTS_DIR = "results/creditcard"

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

PERFORMANCE_DIR = os.path.join(
    RESULTS_DIR,
    "performance"
)

DRIFT_DIR = os.path.join(
    RESULTS_DIR,
    "drift"
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_DIR, exist_ok=True)
os.makedirs(DRIFT_DIR, exist_ok=True)


SUMMARY_PATH = os.path.join(
    METRICS_DIR,
    "creditcard_adaptive_adwin_reset_summary.csv"
)

PERFORMANCE_PATH = os.path.join(
    PERFORMANCE_DIR,
    "creditcard_adaptive_adwin_reset_performance_history.csv"
)

DRIFT_PATH = os.path.join(
    DRIFT_DIR,
    "creditcard_adwin_reset_drift_events.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 5000

ADWIN_DELTA = 0.002


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/8] Loading processed Credit Card dataset...")

data = pd.read_csv(
    INPUT_PATH
)

print(
    "Dataset shape:",
    data.shape
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print(
    "\n[2/8] Sorting transactions chronologically..."
)

data = data.sort_values(
    by=[
        "TransactionDT",
        "TransactionID"
    ]
).reset_index(
    drop=True
)

print(
    "Transactions:",
    len(data)
)


# ============================================================
# TARGET
# ============================================================

TARGET = "isFraud"


# ============================================================
# FEATURES
# ============================================================

NON_FEATURE_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    TARGET
}

FEATURE_COLUMNS = [
    column
    for column in data.columns
    if column not in NON_FEATURE_COLUMNS
]

print(
    "Model features:",
    len(FEATURE_COLUMNS)
)


# ============================================================
# FEATURE BUILDER
# ============================================================

def build_features(row):

    x = {}

    for feature in FEATURE_COLUMNS:

        value = row[feature]

        if pd.isna(value):
            value = 0.0

        try:
            value = float(value)

        except (
            ValueError,
            TypeError
        ):
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        x[feature] = value

    return x


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_positive,
    true_negative,
    false_positive,
    false_negative
):

    total = (
        true_positive
        +
        true_negative
        +
        false_positive
        +
        false_negative
    )

    accuracy = (
        (
            true_positive
            +
            true_negative
        )
        /
        total
        if total > 0
        else 0.0
    )

    precision_denominator = (
        true_positive
        +
        false_positive
    )

    precision = (
        true_positive
        /
        precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall_denominator = (
        true_positive
        +
        false_negative
    )

    recall = (
        true_positive
        /
        recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = (
        precision
        +
        recall
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    fpr_denominator = (
        true_negative
        +
        false_positive
    )

    fpr = (
        false_positive
        /
        fpr_denominator
        if fpr_denominator > 0
        else 0.0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        fpr
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model():

    return (
        preprocessing.StandardScaler()
        |
        linear_model.LogisticRegression()
    )


# ============================================================
# MODEL + ADWIN
# ============================================================

print(
    "\n[3/8] Creating ADWIN adaptive model..."
)

model = create_model()

adwin = drift.ADWIN(
    delta=ADWIN_DELTA
)


# ============================================================
# STORAGE
# ============================================================

performance_history = []

drift_events = []

total_tp = 0
total_tn = 0
total_fp = 0
total_fn = 0

drift_count = 0

window_tp = 0
window_tn = 0
window_fp = 0
window_fn = 0

window_start_transaction = 1

start_time = time.time()


# ============================================================
# STREAM PROCESSING
# ============================================================

print(
    "\n[4/8] Processing chronological transaction stream..."
)


for index, row in data.iterrows():

    transaction_number = index + 1

    x = build_features(row)

    y = int(row[TARGET])


    # --------------------------------------------------------
    # PREDICT FIRST
    # --------------------------------------------------------

    probabilities = model.predict_proba_one(x)

    fraud_probability = probabilities.get(
        1,
        0.0
    )

    prediction = int(
        fraud_probability >= 0.5
    )


    # --------------------------------------------------------
    # UPDATE GLOBAL CONFUSION MATRIX
    # --------------------------------------------------------

    if y == 1 and prediction == 1:

        total_tp += 1
        window_tp += 1

    elif y == 0 and prediction == 0:

        total_tn += 1
        window_tn += 1

    elif y == 0 and prediction == 1:

        total_fp += 1
        window_fp += 1

    elif y == 1 and prediction == 0:

        total_fn += 1
        window_fn += 1


    # --------------------------------------------------------
    # ADWIN UPDATE
    #
    # Error = 1 when prediction is wrong
    # Error = 0 when prediction is correct
    # --------------------------------------------------------

    error = int(
        prediction != y
    )

    adwin.update(error)


    # --------------------------------------------------------
    # DRIFT DETECTION
    # --------------------------------------------------------

    if adwin.drift_detected:

        drift_count += 1

        drift_event = {

            "drift_number":
                drift_count,

            "transaction":
                transaction_number,

            "TransactionID":
                row["TransactionID"],

            "TransactionDT":
                row["TransactionDT"],

            "adwin_width":
                adwin.width,

            "fraud_probability":
                fraud_probability,

            "prediction":
                prediction,

            "actual":
                y,

            "error":
                error

        }

        drift_events.append(
            drift_event
        )


        print(
            "\n" + "!" * 70
        )

        print(
            "CONCEPT DRIFT DETECTED"
        )

        print(
            "Transaction:",
            transaction_number
        )

        print(
            "TransactionDT:",
            row["TransactionDT"]
        )

        print(
            "Drift number:",
            drift_count
        )

        print(
            "ADWIN width:",
            adwin.width
        )

        print(
            "Fraud probability:",
            f"{fraud_probability:.6f}"
        )

        print(
            "Adapting model..."
        )


        # ----------------------------------------------------
        # RESET MODEL
        # ----------------------------------------------------

        model = create_model()

        print(
            "Model reset completed."
        )


    # --------------------------------------------------------
    # LEARN AFTER PREDICTION
    # --------------------------------------------------------

    model.learn_one(
        x,
        y
    )


    # ========================================================
    # WINDOW EVALUATION
    # ========================================================

    if (
        transaction_number % WINDOW_SIZE == 0
        or
        transaction_number == len(data)
    ):

        window_end_transaction = transaction_number

        (
            accuracy,
            precision,
            recall,
            f1,
            fpr
        ) = calculate_metrics(
            window_tp,
            window_tn,
            window_fp,
            window_fn
        )


        performance_history.append({

            "window":
                len(performance_history) + 1,

            "window_start":
                window_start_transaction,

            "window_end":
                window_end_transaction,

            "transactions_processed":
                window_end_transaction,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "fpr":
                fpr,

            "true_positive":
                window_tp,

            "true_negative":
                window_tn,

            "false_positive":
                window_fp,

            "false_negative":
                window_fn,

            "drift_events":
                drift_count,

            "adwin_width":
                adwin.width
        })


        print(
            f"\nWindow {len(performance_history)}"
        )

        print(
            "-" * 50
        )

        print(
            "Transactions:",
            window_start_transaction,
            "-",
            window_end_transaction
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )

        print(
            f"FPR      : {fpr:.4f}"
        )

        print(
            "Drifts   :",
            drift_count
        )


        # Reset window counters

        window_tp = 0
        window_tn = 0
        window_fp = 0
        window_fn = 0

        window_start_transaction = (
            transaction_number + 1
        )


# ============================================================
# FINAL METRICS
# ============================================================

print(
    "\n[5/8] Calculating final metrics..."
)

(
    accuracy,
    precision,
    recall,
    f1,
    fpr
) = calculate_metrics(
    total_tp,
    total_tn,
    total_fp,
    total_fn
)


# ============================================================
# KAPPA
# ============================================================

total_transactions = len(data)

actual_positive = (
    total_tp
    +
    total_fn
)

actual_negative = (
    total_tn
    +
    total_fp
)

predicted_positive = (
    total_tp
    +
    total_fp
)

predicted_negative = (
    total_tn
    +
    total_fn
)

observed_accuracy = accuracy

expected_accuracy = (

    (
        actual_positive
        *
        predicted_positive
    )
    +
    (
        actual_negative
        *
        predicted_negative
    )

) / (
    total_transactions
    *
    total_transactions
)

kappa_denominator = (
    1
    -
    expected_accuracy
)

kappa = (

    (
        observed_accuracy
        -
        expected_accuracy
    )
    /
    kappa_denominator

    if kappa_denominator != 0

    else 0.0
)


# ============================================================
# THROUGHPUT
# ============================================================

elapsed_time = (
    time.time()
    -
    start_time
)

throughput = (

    total_transactions
    /
    elapsed_time

    if elapsed_time > 0

    else 0.0
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n[6/8] Creating final summary..."
)


summary = pd.DataFrame([{

    "model":
        "ADWIN + Reset LR",

    "transactions":
        total_transactions,

    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1":
        f1,

    "kappa":
        kappa,

    "fpr":
        fpr,

    "drift_events":
        drift_count,

    "true_positive":
        total_tp,

    "true_negative":
        total_tn,

    "false_positive":
        total_fp,

    "false_negative":
        total_fn,

    "throughput_transactions_per_second":
        throughput,

    "processing_time_seconds":
        elapsed_time,

    "adwin_delta":
        ADWIN_DELTA,

    "window_size":
        WINDOW_SIZE

}])


# ============================================================
# SAVE RESULTS
# ============================================================

print(
    "\n[7/8] Saving results..."
)


summary.to_csv(
    SUMMARY_PATH,
    index=False
)


pd.DataFrame(
    performance_history
).to_csv(
    PERFORMANCE_PATH,
    index=False
)


pd.DataFrame(
    drift_events
).to_csv(
    DRIFT_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CREDIT CARD ADWIN + RESET COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nFinal Results"
)

print(
    "-" * 50
)

print(
    f"Transactions : {total_transactions}"
)

print(
    f"Accuracy     : {accuracy:.4f}"
)

print(
    f"Precision    : {precision:.4f}"
)

print(
    f"Recall       : {recall:.4f}"
)

print(
    f"F1 Score     : {f1:.4f}"
)

print(
    f"Kappa        : {kappa:.4f}"
)

print(
    f"FPR          : {fpr:.4f}"
)

print(
    f"Drift events : {drift_count}"
)

print(
    f"Throughput   : {throughput:.2f} tx/sec"
)

print(
    f"Processing   : {elapsed_time:.2f} sec"
)


print(
    "\nConfusion Matrix"
)

print(
    "-" * 50
)

print(
    f"True Positive : {total_tp}"
)

print(
    f"True Negative : {total_tn}"
)

print(
    f"False Positive: {total_fp}"
)

print(
    f"False Negative: {total_fn}"
)


print(
    "\nFiles saved:"
)

print(
    SUMMARY_PATH
)

print(
    PERFORMANCE_PATH
)

print(
    DRIFT_PATH
)


print(
    "\nDone."
)
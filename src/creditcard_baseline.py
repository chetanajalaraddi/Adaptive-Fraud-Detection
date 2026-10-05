import os
import time
import numpy as np
import pandas as pd

from river import linear_model
from river import preprocessing


# ============================================================
# STEP 15 - CREDIT CARD PREQUENTIAL BASELINE
# ============================================================

print("=" * 70)
print("STEP 15 - CREDIT CARD PREQUENTIAL BASELINE MODEL")
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

os.makedirs(
    METRICS_DIR,
    exist_ok=True
)

os.makedirs(
    PERFORMANCE_DIR,
    exist_ok=True
)


SUMMARY_PATH = os.path.join(
    METRICS_DIR,
    "creditcard_baseline_online_logistic_summary.csv"
)

METRICS_PATH = os.path.join(
    METRICS_DIR,
    "creditcard_baseline_online_logistic.csv"
)

HISTORY_PATH = os.path.join(
    PERFORMANCE_DIR,
    "creditcard_baseline_performance_history.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "isFraud"

WINDOW_SIZE = 5000


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
    "\n[2/8] Verifying chronological transaction stream..."
)

data = data.sort_values(
    by=[
        "TransactionDT",
        "TransactionID"
    ],
    kind="stable"
).reset_index(
    drop=True
)

print(
    "Transactions:",
    f"{len(data):,}"
)

print(
    "Chronological order:",
    "OK"
    if data["TransactionDT"].is_monotonic_increasing
    else "FAILED"
)


if not data["TransactionDT"].is_monotonic_increasing:

    raise ValueError(
        "Transactions are not chronologically ordered."
    )


# ============================================================
# TARGET VALIDATION
# ============================================================

if TARGET not in data.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# FEATURE COLUMNS
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
# TARGET DISTRIBUTION
# ============================================================

fraud_count = int(
    data[TARGET].sum()
)

normal_count = (
    len(data)
    -
    fraud_count
)


print(
    "\nClass distribution:"
)

print(
    "Normal:",
    f"{normal_count:,}"
)

print(
    "Fraud :",
    f"{fraud_count:,}"
)

print(
    "Fraud %:",
    f"{fraud_count / len(data) * 100:.4f}%"
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
# MODEL
# ============================================================

print(
    "\n[3/8] Creating online logistic regression..."
)

model = (
    preprocessing.StandardScaler()
    |
    linear_model.LogisticRegression()
)


# ============================================================
# PERFORMANCE STORAGE
# ============================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


all_labels = []
all_probabilities = []


performance_history = []


# ============================================================
# WINDOW METRICS
# ============================================================

window_labels = []
window_probabilities = []


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)


    tp = int(
        np.sum(
            (y_true == 1)
            &
            (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (predictions == 0)
        )
    )


    total = (
        tp
        +
        tn
        +
        fp
        +
        fn
    )


    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    f1 = (
        2
        *
        precision
        *
        recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }


# ============================================================
# STREAM PROCESSING
# ============================================================

print(
    "\n[4/8] Running prequential evaluation..."
)

print(
    "Evaluation rule: PREDICT -> EVALUATE -> LEARN"
)

print(
    "Decision threshold: 0.50"
)


start_time = time.time()


for index, row in data.iterrows():

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    x = build_features(
        row
    )

    y = int(
        row[TARGET]
    )


    # --------------------------------------------------------
    # PREDICT FIRST
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba_one(
            x
        )
    )

    fraud_probability = float(
        probabilities.get(
            1,
            0.0
        )
    )


    if not np.isfinite(
        fraud_probability
    ):

        fraud_probability = 0.0


    # --------------------------------------------------------
    # STORE PREDICTION
    # --------------------------------------------------------

    all_labels.append(
        y
    )

    all_probabilities.append(
        fraud_probability
    )


    window_labels.append(
        y
    )

    window_probabilities.append(
        fraud_probability
    )


    # --------------------------------------------------------
    # EVALUATE BEFORE LEARNING
    # --------------------------------------------------------

    prediction = int(
        fraud_probability >= 0.5
    )


    if y == 1 and prediction == 1:

        true_positive += 1

    elif y == 0 and prediction == 0:

        true_negative += 1

    elif y == 0 and prediction == 1:

        false_positive += 1

    elif y == 1 and prediction == 0:

        false_negative += 1


    # --------------------------------------------------------
    # LEARN AFTER PREDICTION
    # --------------------------------------------------------

    model.learn_one(
        x,
        y
    )


    transaction_number = (
        index + 1
    )


    # --------------------------------------------------------
    # WINDOW END
    # --------------------------------------------------------

    if (
        transaction_number % WINDOW_SIZE == 0
        or
        transaction_number == len(data)
    ):

        window_metric = calculate_metrics(
            window_labels,
            window_probabilities,
            threshold=0.5
        )


        window_start = (
            transaction_number
            -
            len(window_labels)
            +
            1
        )


        window_end = (
            transaction_number
        )


        performance_history.append({

            "window":
                len(performance_history) + 1,

            "window_start":
                window_start,

            "window_end":
                window_end,

            "transactions_processed":
                transaction_number,

            "accuracy":
                window_metric["accuracy"],

            "precision":
                window_metric["precision"],

            "recall":
                window_metric["recall"],

            "f1":
                window_metric["f1"],

            "fpr":
                window_metric["fpr"],

            "true_positive":
                window_metric["true_positive"],

            "true_negative":
                window_metric["true_negative"],

            "false_positive":
                window_metric["false_positive"],

            "false_negative":
                window_metric["false_negative"]

        })


        print(
            f"\nWindow "
            f"{len(performance_history)}"
        )

        print(
            "-" * 50
        )

        print(
            "Transactions:",
            f"{window_start:,}",
            "-",
            f"{window_end:,}"
        )

        print(
            f"Accuracy : "
            f"{window_metric['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{window_metric['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{window_metric['recall']:.4f}"
        )

        print(
            f"F1       : "
            f"{window_metric['f1']:.4f}"
        )

        print(
            f"FPR      : "
            f"{window_metric['fpr']:.4f}"
        )


        window_labels = []
        window_probabilities = []


# ============================================================
# PROCESSING TIME
# ============================================================

processing_time = (
    time.time()
    -
    start_time
)


# ============================================================
# FINAL METRICS
# ============================================================

print(
    "\n[5/8] Calculating final metrics..."
)


total_transactions = len(
    data
)


accuracy = (
    true_positive
    +
    true_negative
) / total_transactions


precision = (
    true_positive
    /
    (
        true_positive
        +
        false_positive
    )
    if (
        true_positive
        +
        false_positive
    ) > 0
    else 0.0
)


recall = (
    true_positive
    /
    (
        true_positive
        +
        false_negative
    )
    if (
        true_positive
        +
        false_negative
    ) > 0
    else 0.0
)


f1 = (
    2
    *
    precision
    *
    recall
    /
    (
        precision
        +
        recall
    )
    if (
        precision
        +
        recall
    ) > 0
    else 0.0
)


fpr = (
    false_positive
    /
    (
        false_positive
        +
        true_negative
    )
    if (
        false_positive
        +
        true_negative
    ) > 0
    else 0.0
)


# ============================================================
# KAPPA
# ============================================================

observed_accuracy = accuracy


actual_positive = (
    true_positive
    +
    false_negative
)

actual_negative = (
    true_negative
    +
    false_positive
)

predicted_positive = (
    true_positive
    +
    false_positive
)

predicted_negative = (
    true_negative
    +
    false_negative
)


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

throughput = (
    total_transactions
    /
    processing_time
    if processing_time > 0
    else 0.0
)


# ============================================================
# [6/8] CREATING SUMMARY
# ============================================================

print(
    "\n[6/8] Creating final summary..."
)


summary = pd.DataFrame([{

    "model":
        "Credit Card Online Logistic Regression",

    "dataset":
        "Credit Card Fraud Detection",

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

    "true_positive":
        true_positive,

    "true_negative":
        true_negative,

    "false_positive":
        false_positive,

    "false_negative":
        false_negative,

    "throughput":
        throughput,

    "processing_time_seconds":
        processing_time

}])


# ============================================================
# [7/8] SAVING RESULTS
# ============================================================

print(
    "\n[7/8] Saving results..."
)


summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ------------------------------------------------------------
# SAVE WINDOW PERFORMANCE
# ------------------------------------------------------------

history = pd.DataFrame(
    performance_history
)


history.insert(
    0,
    "model",
    "Credit Card Online Logistic Regression"
)


history.to_csv(
    HISTORY_PATH,
    index=False
)


# ------------------------------------------------------------
# SAVE DETAILED METRICS
# ------------------------------------------------------------

metrics = pd.DataFrame([{

    "model":
        "Credit Card Online Logistic Regression",

    "transactions_processed":
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

    "true_positive":
        true_positive,

    "true_negative":
        true_negative,

    "false_positive":
        false_positive,

    "false_negative":
        false_negative,

    "throughput_transactions_per_second":
        throughput,

    "processing_time_seconds":
        processing_time

}])


metrics.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# [8/8] FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CREDIT CARD PREQUENTIAL BASELINE COMPLETE"
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
    f"Transactions : "
    f"{total_transactions:,}"
)

print(
    f"Accuracy     : "
    f"{accuracy:.4f}"
)

print(
    f"Precision    : "
    f"{precision:.4f}"
)

print(
    f"Recall       : "
    f"{recall:.4f}"
)

print(
    f"F1 Score     : "
    f"{f1:.4f}"
)

print(
    f"Kappa        : "
    f"{kappa:.4f}"
)

print(
    f"FPR          : "
    f"{fpr:.4f}"
)

print(
    f"Throughput   : "
    f"{throughput:.2f} tx/sec"
)

print(
    f"Processing   : "
    f"{processing_time:.2f} sec"
)


print(
    "\nConfusion Matrix"
)

print(
    "-" * 50
)

print(
    "True Positive :",
    true_positive
)

print(
    "True Negative :",
    true_negative
)

print(
    "False Positive:",
    false_positive
)

print(
    "False Negative:",
    false_negative
)


print(
    "\nFiles saved:"
)

print(
    METRICS_PATH
)

print(
    SUMMARY_PATH
)

print(
    HISTORY_PATH
)


print(
    "\nDone."
)
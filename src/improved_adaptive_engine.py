import os
import time
import csv
from collections import deque

import pandas as pd
import numpy as np

from river import linear_model
from river import preprocessing
from river import drift
from river import metrics


# ============================================================
# STEP 7 - IMPROVED DRIFT-AWARE ADAPTIVE MODEL
# ============================================================

print("=" * 70)
print("STEP 7 - IMPROVED DRIFT-AWARE ADAPTIVE MODEL")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
    "data/processed/ieee_features.csv"
)

RESULTS_DIR = "results"

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

DRIFT_DIR = os.path.join(
    RESULTS_DIR,
    "drift"
)

OUTPUT_PATH = os.path.join(
    METRICS_DIR,
    "improved_adaptive_logistic.csv"
)

SUMMARY_PATH = os.path.join(
    METRICS_DIR,
    "improved_adaptive_logistic_summary.csv"
)

DRIFT_PATH = os.path.join(
    DRIFT_DIR,
    "improved_adaptive_drift_events.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TRANSACTIONS = None

EVALUATION_INTERVAL = 10000

# Number of recent transactions retained
REPLAY_BUFFER_SIZE = 2000

# Number of recent transactions replayed
# after a detected drift
REPLAY_SAMPLES = 500


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    METRICS_DIR,
    exist_ok=True
)

os.makedirs(
    DRIFT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/9] Loading feature dataset...")

if not os.path.exists(INPUT_PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_PATH}"
    )

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
    "\n[2/9] Sorting transaction stream..."
)

data = data.sort_values(
    by=[
        "TransactionDT",
        "TransactionID"
    ]
).reset_index(
    drop=True
)


if MAX_TRANSACTIONS is not None:

    data = data.iloc[
        :MAX_TRANSACTIONS
    ].copy()


print(
    "Transactions to process:",
    len(data)
)


# ============================================================
# TARGET
# ============================================================

TARGET = "isFraud"


if TARGET not in data.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# MODEL FEATURES
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
    "Number of model features:",
    len(FEATURE_COLUMNS)
)


# ============================================================
# CREATE ONLINE MODEL
# ============================================================

print(
    "\n[3/9] Creating online Logistic Regression..."
)

model = (
    preprocessing.StandardScaler()
    |
    linear_model.LogisticRegression()
)


# ============================================================
# ADWIN DRIFT DETECTOR
# ============================================================

print(
    "[4/9] Creating ADWIN drift detector..."
)

adwin = drift.ADWIN()


# ============================================================
# METRICS
# ============================================================

print(
    "[5/9] Initializing evaluation metrics..."
)

accuracy = metrics.Accuracy()

precision = metrics.Precision()

recall = metrics.Recall()

f1 = metrics.F1()

roc_auc = metrics.ROCAUC()

kappa = metrics.CohenKappa()


# ============================================================
# PR-AUC STORAGE
# ============================================================

all_labels = []

all_probabilities = []


# ============================================================
# CONFUSION MATRIX
# ============================================================

true_positive = 0

true_negative = 0

false_positive = 0

false_negative = 0


# ============================================================
# PERFORMANCE HISTORY
# ============================================================

performance_history = []


# ============================================================
# RECENT HISTORY BUFFER
# ============================================================

replay_buffer = deque(
    maxlen=REPLAY_BUFFER_SIZE
)


# ============================================================
# DRIFT COUNT
# ============================================================

drift_count = 0


# ============================================================
# INITIALIZE OUTPUT FILES
# ============================================================

metrics_header = [

    "transactions_processed",

    "accuracy",

    "precision",

    "recall",

    "f1",

    "roc_auc",

    "kappa",

    "true_positive",

    "true_negative",

    "false_positive",

    "false_negative",

    "throughput_transactions_per_second"

]


drift_header = [

    "transaction_index",

    "TransactionDT",

    "drift_number",

    "accuracy_at_detection",

    "precision_at_detection",

    "recall_at_detection",

    "f1_at_detection",

    "roc_auc_at_detection",

    "kappa_at_detection",

    "adwin_width",

    "replay_samples"

]


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        metrics_header
    )


with open(
    DRIFT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        drift_header
    )


# ============================================================
# STREAM PROCESSING
# ============================================================

print(
    "\n[6/9] Starting chronological stream..."
)

print(
    "Protocol: Predict -> Evaluate -> Adapt -> Learn"
)

start_time = time.time()


for index, row in data.iterrows():

    # ========================================================
    # BUILD FEATURE DICTIONARY
    # ========================================================

    x = {}

    for feature in FEATURE_COLUMNS:

        value = row[feature]

        # Same preprocessing as streaming_engine.py

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


    # ========================================================
    # TRUE LABEL
    # ========================================================

    y = int(
        row[TARGET]
    )


    # ========================================================
    # PREDICT
    # ========================================================

    prediction = model.predict_one(x)

    probabilities = model.predict_proba_one(x)

    fraud_probability = probabilities.get(
        1,
        0.0
    )


    # ========================================================
    # EVALUATE
    # ========================================================

    accuracy.update(
        y,
        prediction
    )

    precision.update(
        y,
        prediction
    )

    recall.update(
        y,
        prediction
    )

    f1.update(
        y,
        prediction
    )

    roc_auc.update(
        y,
        fraud_probability
    )

    kappa.update(
        y,
        prediction
    )


    # ========================================================
    # STORE PR-AUC VALUES
    # ========================================================

    all_labels.append(
        y
    )

    all_probabilities.append(
        fraud_probability
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    if (
        y == 1
        and
        prediction == 1
    ):

        true_positive += 1

    elif (
        y == 0
        and
        prediction == 0
    ):

        true_negative += 1

    elif (
        y == 0
        and
        prediction == 1
    ):

        false_positive += 1

    elif (
        y == 1
        and
        prediction == 0
    ):

        false_negative += 1


    # ========================================================
    # ADWIN
    # ========================================================

    error = int(
        prediction != y
    )

    adwin.update(
        error
    )


    # ========================================================
    # DRIFT DETECTED
    # ========================================================

    if adwin.drift_detected:

        drift_count += 1

        print(
            "\n"
            + "!" * 70
        )

        print(
            "CONCEPT DRIFT DETECTED"
        )

        print(
            f"Transaction: {index}"
        )

        print(
            f"TransactionDT: "
            f"{row['TransactionDT']}"
        )

        print(
            f"Drift number: "
            f"{drift_count}"
        )

        print(
            f"ADWIN width: "
            f"{adwin.width}"
        )

        print(
            f"F1: "
            f"{f1.get():.4f}"
        )

        print(
            f"Recall: "
            f"{recall.get():.4f}"
        )

        print(
            "Applying recent-history adaptation..."
        )


        # ====================================================
        # GET PREVIOUS RECENT HISTORY
        # ====================================================

        recent_history = list(
            replay_buffer
        )


        if len(recent_history) > REPLAY_SAMPLES:

            recent_history = recent_history[
                -REPLAY_SAMPLES:
            ]


        # ====================================================
        # REPLAY RECENT TRANSACTIONS
        # ====================================================

        for replay_x, replay_y in recent_history:

            model.learn_one(
                replay_x,
                replay_y
            )


        print(
            f"Replayed "
            f"{len(recent_history)} "
            f"recent transactions."
        )

        print(
            "Existing model knowledge retained."
        )


        # ====================================================
        # SAVE DRIFT EVENT
        # ====================================================

        with open(
            DRIFT_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([

                index,

                row["TransactionDT"],

                drift_count,

                accuracy.get(),

                precision.get(),

                recall.get(),

                f1.get(),

                roc_auc.get(),

                kappa.get(),

                adwin.width,

                len(recent_history)

            ])


    # ========================================================
    # NORMAL ONLINE LEARNING
    # ========================================================

    model.learn_one(
        x,
        y
    )


    # ========================================================
    # ADD CURRENT TRANSACTION TO REPLAY BUFFER
    # ========================================================

    replay_buffer.append(
        (
            x.copy(),
            y
        )
    )


    # ========================================================
    # PERIODIC PERFORMANCE
    # ========================================================

    transaction_number = (
        index + 1
    )


    if (
        transaction_number
        %
        EVALUATION_INTERVAL
        == 0
        or
        transaction_number
        ==
        len(data)
    ):

        elapsed = (
            time.time()
            -
            start_time
        )

        throughput = (

            transaction_number
            /
            elapsed

            if elapsed > 0

            else 0

        )


        performance_history.append({

            "transactions_processed":
                transaction_number,

            "accuracy":
                accuracy.get(),

            "precision":
                precision.get(),

            "recall":
                recall.get(),

            "f1":
                f1.get(),

            "roc_auc":
                roc_auc.get(),

            "kappa":
                kappa.get(),

            "true_positive":
                true_positive,

            "true_negative":
                true_negative,

            "false_positive":
                false_positive,

            "false_negative":
                false_negative,

            "throughput_transactions_per_second":
                throughput

        })


        print(
            f"\nProcessed: "
            f"{transaction_number:,}"
        )

        print(
            f"Accuracy : "
            f"{accuracy.get():.4f}"
        )

        print(
            f"Precision: "
            f"{precision.get():.4f}"
        )

        print(
            f"Recall   : "
            f"{recall.get():.4f}"
        )

        print(
            f"F1       : "
            f"{f1.get():.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{roc_auc.get():.4f}"
        )

        print(
            f"Kappa    : "
            f"{kappa.get():.4f}"
        )

        print(
            f"Drifts   : "
            f"{drift_count}"
        )

        print(
            f"Throughput: "
            f"{throughput:.2f} tx/sec"
        )


# ============================================================
# PR-AUC
# ============================================================

print(
    "\n[7/9] Calculating PR-AUC..."
)


def calculate_pr_auc(
    y_true,
    y_scores
):

    y_true = np.asarray(
        y_true
    )

    y_scores = np.asarray(
        y_scores
    )


    if len(
        np.unique(y_true)
    ) < 2:

        return 0.0


    order = np.argsort(
        -y_scores
    )


    y_true = y_true[
        order
    ]


    cumulative_tp = np.cumsum(
        y_true
    )


    cumulative_fp = np.cumsum(
        1 - y_true
    )


    total_positive = np.sum(
        y_true
    )


    if total_positive == 0:

        return 0.0


    recall_values = (

        cumulative_tp
        /
        total_positive

    )


    precision_values = (

        cumulative_tp
        /
        (
            cumulative_tp
            +
            cumulative_fp
        )

    )


    recall_values = np.concatenate(
        [
            [0.0],
            recall_values
        ]
    )


    precision_values = np.concatenate(
        [
            [1.0],
            precision_values
        ]
    )


    try:

        pr_auc_value = np.trapezoid(
            precision_values,
            recall_values
        )

    except AttributeError:

        pr_auc_value = np.trapz(
            precision_values,
            recall_values
        )


    return float(
        pr_auc_value
    )


final_pr_auc = calculate_pr_auc(
    all_labels,
    all_probabilities
)


# ============================================================
# SAVE PERFORMANCE HISTORY
# ============================================================

print(
    "\n[8/9] Saving performance history..."
)


results = pd.DataFrame(
    performance_history
)


results["pr_auc"] = np.nan


if len(results) > 0:

    results.loc[
        results.index[-1],
        "pr_auc"
    ] = final_pr_auc


results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL PERFORMANCE
# ============================================================

elapsed_time = (
    time.time()
    -
    start_time
)


throughput = (

    len(data)
    /
    elapsed_time

    if elapsed_time > 0

    else 0

)


# ============================================================
# FALSE POSITIVE RATE
# ============================================================

if (
    true_negative
    +
    false_positive
) > 0:

    fpr = (

        false_positive
        /
        (
            true_negative
            +
            false_positive
        )

    )

else:

    fpr = 0.0


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "model":
        "Improved Adaptive LR + ADWIN + Recent Replay",

    "transactions":
        len(data),

    "accuracy":
        accuracy.get(),

    "precision":
        precision.get(),

    "recall":
        recall.get(),

    "f1":
        f1.get(),

    "roc_auc":
        roc_auc.get(),

    "pr_auc":
        final_pr_auc,

    "kappa":
        kappa.get(),

    "fpr":
        fpr,

    "drift_events":
        drift_count,

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
        elapsed_time

}])


summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n[9/9] IMPROVED ADAPTIVE EXPERIMENT COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Transactions : "
    f"{len(data):,}"
)

print(
    f"Accuracy     : "
    f"{accuracy.get():.4f}"
)

print(
    f"Precision    : "
    f"{precision.get():.4f}"
)

print(
    f"Recall       : "
    f"{recall.get():.4f}"
)

print(
    f"F1 Score     : "
    f"{f1.get():.4f}"
)

print(
    f"ROC-AUC      : "
    f"{roc_auc.get():.4f}"
)

print(
    f"PR-AUC       : "
    f"{final_pr_auc:.4f}"
)

print(
    f"Kappa        : "
    f"{kappa.get():.4f}"
)

print(
    f"FPR          : "
    f"{fpr:.4f}"
)

print(
    f"Drift Events : "
    f"{drift_count}"
)

print(
    f"Throughput   : "
    f"{throughput:.2f} tx/sec"
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
    "\nResults saved:"
)

print(
    OUTPUT_PATH
)

print(
    SUMMARY_PATH
)

print(
    DRIFT_PATH
)

print(
    "\nDone."
)
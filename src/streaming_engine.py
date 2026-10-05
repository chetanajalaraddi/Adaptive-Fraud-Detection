import os
import time
import pandas as pd
import numpy as np

from river import linear_model
from river import preprocessing
from river import metrics


# ============================================================
# STEP 5 - CHRONOLOGICAL STREAMING ENGINE
# ============================================================

print("=" * 70)
print("STEP 5 - CHRONOLOGICAL STREAMING ENGINE")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/processed/ieee_features.csv"

RESULTS_DIR = "results/metrics"

os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "baseline_online_logistic.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "baseline_online_logistic_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TRANSACTIONS = None

EVALUATION_INTERVAL = 10000


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/7] Loading feature dataset...")

data = pd.read_csv(INPUT_PATH)

print("Dataset shape:", data.shape)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print("\n[2/7] Sorting transaction stream...")

data = data.sort_values(
    by=["TransactionDT", "TransactionID"]
).reset_index(drop=True)


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
# ONLINE LOGISTIC REGRESSION
# ============================================================

print("\n[3/7] Creating online Logistic Regression model...")


# IMPORTANT:
# Do not pass loss="log_loss" as a string.
# The installed River version expects its default loss
# configuration.

model = (
    preprocessing.StandardScaler()
    |
    linear_model.LogisticRegression()
)


# ============================================================
# METRICS
# ============================================================

print("\n[4/7] Initializing evaluation metrics...")


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
# STREAM PROCESSING
# ============================================================

print("\n[5/7] Starting chronological stream...")

start_time = time.time()


for index, row in data.iterrows():

    # ========================================================
    # BUILD FEATURE DICTIONARY
    # ========================================================

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


    # ========================================================
    # TRUE LABEL
    # ========================================================

    y = int(row[TARGET])


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
    # STORE PROBABILITY
    # ========================================================

    all_labels.append(y)

    all_probabilities.append(
        fraud_probability
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    if y == 1 and prediction == 1:

        true_positive += 1

    elif y == 0 and prediction == 0:

        true_negative += 1

    elif y == 0 and prediction == 1:

        false_positive += 1

    elif y == 1 and prediction == 0:

        false_negative += 1


    # ========================================================
    # LEARN AFTER PREDICTION
    # ========================================================

    model.learn_one(
        x,
        y
    )


    # ========================================================
    # PERIODIC RESULTS
    # ========================================================

    transaction_number = index + 1


    if (
        transaction_number % EVALUATION_INTERVAL == 0
        or transaction_number == len(data)
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
            f"\nProcessed: {transaction_number:,}"
        )

        print(
            f"Accuracy : {accuracy.get():.4f}"
        )

        print(
            f"Precision: {precision.get():.4f}"
        )

        print(
            f"Recall   : {recall.get():.4f}"
        )

        print(
            f"F1       : {f1.get():.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc.get():.4f}"
        )

        print(
            f"Kappa    : {kappa.get():.4f}"
        )

        print(
            f"Throughput: {throughput:.2f} tx/sec"
        )


# ============================================================
# PR-AUC CALCULATION
# ============================================================

print("\nCalculating PR-AUC...")


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


    # Sort scores descending

    order = np.argsort(
        -y_scores
    )

    y_true = y_true[
        order
    ]


    # Cumulative positives

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


    # Compatible with older NumPy versions

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

print("\n[6/7] Saving performance history...")

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
# FINAL METRICS
# ============================================================

elapsed_time = (
    time.time()
    -
    start_time
)


print("\n" + "=" * 70)

print(
    "BASELINE STREAMING EXPERIMENT COMPLETE"
)

print("=" * 70)


print("\nFinal Metrics")

print("-" * 50)


print(
    f"Transactions : {len(data):,}"
)

print(
    f"Accuracy     : {accuracy.get():.4f}"
)

print(
    f"Precision    : {precision.get():.4f}"
)

print(
    f"Recall       : {recall.get():.4f}"
)

print(
    f"F1 Score     : {f1.get():.4f}"
)

print(
    f"ROC-AUC      : {roc_auc.get():.4f}"
)

print(
    f"PR-AUC       : {final_pr_auc:.4f}"
)

print(
    f"Kappa        : {kappa.get():.4f}"
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


print(
    f"FPR          : {fpr:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix")

print("-" * 50)


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


# ============================================================
# PERFORMANCE
# ============================================================

throughput = (
    len(data)
    /
    elapsed_time
    if elapsed_time > 0
    else 0
)


print("\nProcessing Performance")

print("-" * 50)


print(
    f"Total time   : {elapsed_time:.2f} seconds"
)

print(
    f"Throughput   : {throughput:.2f} transactions/sec"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "model":
        "Online Logistic Regression",

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

    "true_positive":
        true_positive,

    "true_negative":
        true_negative,

    "false_positive":
        false_positive,

    "false_negative":
        false_negative,

    "throughput":
        throughput

}])


summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n[7/7] Results saved:")

print(
    OUTPUT_PATH
)

print(
    SUMMARY_PATH
)

print("\nDone.")
import os
import time

import numpy as np
import pandas as pd

from river import linear_model
from river import preprocessing
from river import metrics
from river import drift


# ============================================================
# STEP 6 - ADAPTIVE ONLINE FRAUD DETECTION
# ============================================================

print("=" * 70)
print("STEP 6 - ADAPTIVE ONLINE FRAUD DETECTION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/processed/ieee_features.csv"

RESULTS_DIR = "results/metrics"

DRIFT_DIR = "results/drift"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    DRIFT_DIR,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "adaptive_adwin_logistic.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "adaptive_adwin_logistic_summary.csv"
)

DRIFT_PATH = os.path.join(
    DRIFT_DIR,
    "adwin_drift_events.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

EVALUATION_INTERVAL = 10000

TARGET = "isFraud"


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/8] Loading feature dataset...")

data = pd.read_csv(
    INPUT_PATH
)

print(
    "Dataset shape:",
    data.shape
)


# ============================================================
# SORT TRANSACTION STREAM
# ============================================================

print("\n[2/8] Sorting chronological transaction stream...")

data = data.sort_values(
    by=["TransactionDT", "TransactionID"]
).reset_index(
    drop=True
)

print(
    "Transactions:",
    len(data)
)


# ============================================================
# SELECT FEATURES
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
# MODEL FACTORY
# ============================================================

def create_model():

    return (
        preprocessing.StandardScaler()
        |
        linear_model.LogisticRegression()
    )


# ============================================================
# INITIAL MODEL
# ============================================================

print("\n[3/8] Creating adaptive online model...")

model = create_model()


# ============================================================
# ADWIN DRIFT DETECTOR
# ============================================================

print(
    "\n[4/8] Initializing ADWIN drift detector..."
)

adwin = drift.ADWIN()


# ============================================================
# METRICS
# ============================================================

print(
    "\n[5/8] Initializing evaluation metrics..."
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
# DRIFT EVENTS
# ============================================================

drift_events = []

drift_count = 0


# ============================================================
# STREAM PROCESSING
# ============================================================

print(
    "\n[6/8] Starting adaptive chronological stream..."
)

start_time = time.time()


for index, row in data.iterrows():

    # ========================================================
    # FEATURE DICTIONARY
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

    y = int(
        row[TARGET]
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict_one(
        x
    )

    probabilities = model.predict_proba_one(
        x
    )

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
    # STORE PR-AUC DATA
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

    if y == 1 and prediction == 1:

        true_positive += 1

    elif y == 0 and prediction == 0:

        true_negative += 1

    elif y == 0 and prediction == 1:

        false_positive += 1

    elif y == 1 and prediction == 0:

        false_negative += 1


    # ========================================================
    # CONCEPT DRIFT SIGNAL
    # ========================================================

    prediction_error = int(
        prediction != y
    )

    adwin.update(
        prediction_error
    )


    # ========================================================
    # DRIFT DETECTION
    # ========================================================

    if adwin.drift_detected:

        drift_count += 1

        drift_event = {

            "transaction_index":
                index + 1,

            "TransactionDT":
                row["TransactionDT"],

            "drift_number":
                drift_count,

            "accuracy_at_detection":
                accuracy.get(),

            "precision_at_detection":
                precision.get(),

            "recall_at_detection":
                recall.get(),

            "f1_at_detection":
                f1.get(),

            "roc_auc_at_detection":
                roc_auc.get(),

            "adwin_width":
                adwin.width

        }

        drift_events.append(
            drift_event
        )


        print(
            "\n"
            + "!" * 70
        )

        print(
            "CONCEPT DRIFT DETECTED"
        )

        print(
            "Transaction:",
            index + 1
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
            "F1:",
            round(
                f1.get(),
                4
            )
        )

        print(
            "!" * 70
        )


        # ====================================================
        # ADAPTATION
        # ====================================================

        print(
            "Adapting model..."
        )

        model = create_model()

        print(
            "Model reset completed."
        )


        # Reset detector after adaptation
        adwin = drift.ADWIN()


    # ========================================================
    # ONLINE LEARNING
    # ========================================================

    model.learn_one(
        x,
        y
    )


    # ========================================================
    # PERIODIC EVALUATION
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
    "\nCalculating PR-AUC..."
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
# SAVE PERFORMANCE
# ============================================================

print(
    "\n[7/8] Saving adaptive model results..."
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
# SAVE DRIFT EVENTS
# ============================================================

drift_dataframe = pd.DataFrame(
    drift_events
)


if len(drift_dataframe) == 0:

    drift_dataframe = pd.DataFrame(
        columns=[
            "transaction_index",
            "TransactionDT",
            "drift_number",
            "accuracy_at_detection",
            "precision_at_detection",
            "recall_at_detection",
            "f1_at_detection",
            "roc_auc_at_detection",
            "adwin_width"
        ]
    )


drift_dataframe.to_csv(
    DRIFT_PATH,
    index=False
)


# ============================================================
# FINAL RESULTS
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
    "ADAPTIVE STREAMING EXPERIMENT COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nFinal Metrics"
)

print(
    "-" * 50
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
    f"Drift Events : "
    f"{drift_count}"
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
    f"FPR          : "
    f"{fpr:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

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


print(
    "\nProcessing Performance"
)

print(
    "-" * 50
)

print(
    f"Total time : "
    f"{elapsed_time:.2f} seconds"
)

print(
    f"Throughput : "
    f"{throughput:.2f} transactions/sec"
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "model":
        "Adaptive Logistic Regression + ADWIN",

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
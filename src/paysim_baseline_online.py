import os
import time
import numpy as np
import pandas as pd

from river import linear_model
from river import preprocessing


# ============================================================
# STEP 17 - PAYSIM PREQUENTIAL ONLINE BASELINE
# ============================================================

print("=" * 70)
print("STEP 17 - PAYSIM PREQUENTIAL ONLINE LOGISTIC REGRESSION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/processed/paysim_features.csv"

RESULTS_DIR = "results/paysim"

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


METRICS_PATH = os.path.join(
    METRICS_DIR,
    "paysim_baseline_online_logistic.csv"
)

SUMMARY_PATH = os.path.join(
    METRICS_DIR,
    "paysim_baseline_online_logistic_summary.csv"
)

PERFORMANCE_PATH = os.path.join(
    PERFORMANCE_DIR,
    "paysim_baseline_performance_history.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 50000

WINDOW_SIZE = 50000

TARGET = "isFraud"

NON_FEATURE_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    TARGET
}


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# HELPER FUNCTIONS
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

    kappa_denominator = (
        total * total
    )

    if total > 0:

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

        observed_accuracy = (
            true_positive
            +
            true_negative
        ) / total

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
        ) / kappa_denominator

        kappa_denominator_2 = (
            1.0
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
            kappa_denominator_2
            if kappa_denominator_2 != 0
            else 0.0
        )

    else:

        kappa = 0.0

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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "fpr": fpr
    }


# ============================================================
# LOAD DATA INFORMATION
# ============================================================

print(
    "\n[1/8] Checking processed PaySim dataset..."
)

header = pd.read_csv(
    INPUT_PATH,
    nrows=0
)

print(
    "Dataset columns:",
    len(header.columns)
)

print(
    "Model features:",
    len(FEATURE_COLUMNS)
)


missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in header.columns
]

if missing_features:

    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "\n[2/8] Creating chronological online model..."
)

model = (
    preprocessing.StandardScaler()
    |
    linear_model.LogisticRegression()
)


# ============================================================
# STORAGE
# ============================================================

performance_history = []

global_true_positive = 0
global_true_negative = 0
global_false_positive = 0
global_false_negative = 0

total_transactions = 0

window_true_positive = 0
window_true_negative = 0
window_false_positive = 0
window_false_negative = 0

window_start_transaction = 1

window_number = 0

start_time = time.time()


# ============================================================
# STREAM PROCESSING
# ============================================================

print(
    "\n[3/8] Processing chronological transaction stream..."
)

print(
    "Chunk size:",
    CHUNK_SIZE
)

print(
    "Window size:",
    WINDOW_SIZE
)

print(
    "\nEvaluation protocol:"
)

print(
    "PREDICT -> EVALUATE -> LEARN"
)


for chunk in pd.read_csv(
    INPUT_PATH,
    chunksize=CHUNK_SIZE
):

    for row in chunk.itertuples(
        index=False
    ):

        row_dict = row._asdict()

        x = {}

        for feature in FEATURE_COLUMNS:

            value = row_dict[feature]

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

        y = int(
            row_dict[TARGET]
        )


        # ----------------------------------------------------
        # PREDICT BEFORE LEARNING
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba_one(
                x
            )
        )

        fraud_probability = probabilities.get(
            1,
            0.0
        )


        prediction = int(
            fraud_probability >= 0.50
        )


        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        if y == 1 and prediction == 1:

            global_true_positive += 1
            window_true_positive += 1

        elif y == 0 and prediction == 0:

            global_true_negative += 1
            window_true_negative += 1

        elif y == 0 and prediction == 1:

            global_false_positive += 1
            window_false_positive += 1

        elif y == 1 and prediction == 0:

            global_false_negative += 1
            window_false_negative += 1


        total_transactions += 1


        # ----------------------------------------------------
        # LEARN AFTER PREDICTION
        # ----------------------------------------------------

        model.learn_one(
            x,
            y
        )


        # ----------------------------------------------------
        # WINDOW EVALUATION
        # ----------------------------------------------------

        if (
            total_transactions % WINDOW_SIZE == 0
        ):

            window_number += 1

            metrics = calculate_metrics(
                window_true_positive,
                window_true_negative,
                window_false_positive,
                window_false_negative
            )

            window_end_transaction = (
                total_transactions
            )

            elapsed = (
                time.time()
                -
                start_time
            )

            throughput = (
                total_transactions
                /
                elapsed
                if elapsed > 0
                else 0.0
            )

            result = {

                "window":
                    window_number,

                "transactions_processed":
                    total_transactions,

                "window_start":
                    window_start_transaction,

                "window_end":
                    window_end_transaction,

                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1":
                    metrics["f1"],

                "kappa":
                    metrics["kappa"],

                "fpr":
                    metrics["fpr"],

                "true_positive":
                    window_true_positive,

                "true_negative":
                    window_true_negative,

                "false_positive":
                    window_false_positive,

                "false_negative":
                    window_false_negative,

                "throughput_transactions_per_second":
                    throughput
            }

            performance_history.append(
                result
            )


            print(
                f"\nWindow {window_number}"
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
                f"Accuracy : {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {metrics['precision']:.4f}"
            )

            print(
                f"Recall   : {metrics['recall']:.4f}"
            )

            print(
                f"F1       : {metrics['f1']:.4f}"
            )

            print(
                f"FPR      : {metrics['fpr']:.4f}"
            )


            # Reset window counters

            window_true_positive = 0
            window_true_negative = 0
            window_false_positive = 0
            window_false_negative = 0

            window_start_transaction = (
                total_transactions + 1
            )


# ============================================================
# FINAL PARTIAL WINDOW
# ============================================================

if (
    window_start_transaction
    <=
    total_transactions
):

    window_number += 1

    metrics = calculate_metrics(
        window_true_positive,
        window_true_negative,
        window_false_positive,
        window_false_negative
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    throughput = (
        total_transactions
        /
        elapsed
        if elapsed > 0
        else 0.0
    )

    result = {

        "window":
            window_number,

        "transactions_processed":
            total_transactions,

        "window_start":
            window_start_transaction,

        "window_end":
            total_transactions,

        "accuracy":
            metrics["accuracy"],

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1":
            metrics["f1"],

        "kappa":
            metrics["kappa"],

        "fpr":
            metrics["fpr"],

        "true_positive":
            window_true_positive,

        "true_negative":
            window_true_negative,

        "false_positive":
            window_false_positive,

        "false_negative":
            window_false_negative,

        "throughput_transactions_per_second":
            throughput
    }

    performance_history.append(
        result
    )

    print(
        f"\nWindow {window_number}"
    )

    print(
        "-" * 50
    )

    print(
        "Transactions:",
        window_start_transaction,
        "-",
        total_transactions
    )

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    print(
        f"FPR      : {metrics['fpr']:.4f}"
    )


# ============================================================
# FINAL METRICS
# ============================================================

print(
    "\n[5/8] Calculating final metrics..."
)

final_metrics = calculate_metrics(
    global_true_positive,
    global_true_negative,
    global_false_positive,
    global_false_negative
)

processing_time = (
    time.time()
    -
    start_time
)

throughput = (
    total_transactions
    /
    processing_time
    if processing_time > 0
    else 0.0
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n[6/8] Creating final summary..."
)

summary = pd.DataFrame([{

    "model":
        "PaySim Online Logistic Regression",

    "transactions":
        total_transactions,

    "accuracy":
        final_metrics["accuracy"],

    "precision":
        final_metrics["precision"],

    "recall":
        final_metrics["recall"],

    "f1":
        final_metrics["f1"],

    "kappa":
        final_metrics["kappa"],

    "fpr":
        final_metrics["fpr"],

    "true_positive":
        global_true_positive,

    "true_negative":
        global_true_negative,

    "false_positive":
        global_false_positive,

    "false_negative":
        global_false_negative,

    "throughput":
        throughput,

    "processing_time_seconds":
        processing_time
}])


# ============================================================
# SAVE RESULTS
# ============================================================

print(
    "\n[7/8] Saving results..."
)

summary.to_csv(
    METRICS_PATH,
    index=False
)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

performance = pd.DataFrame(
    performance_history
)

performance.to_csv(
    PERFORMANCE_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PAYSIM PREQUENTIAL BASELINE COMPLETE"
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
    f"Transactions : {total_transactions:,}"
)

print(
    f"Accuracy     : {final_metrics['accuracy']:.4f}"
)

print(
    f"Precision    : {final_metrics['precision']:.4f}"
)

print(
    f"Recall       : {final_metrics['recall']:.4f}"
)

print(
    f"F1 Score     : {final_metrics['f1']:.4f}"
)

print(
    f"Kappa        : {final_metrics['kappa']:.4f}"
)

print(
    f"FPR          : {final_metrics['fpr']:.4f}"
)

print(
    f"Throughput   : {throughput:.2f} tx/sec"
)

print(
    f"Processing   : {processing_time:.2f} sec"
)


print(
    "\nConfusion Matrix"
)

print(
    "-" * 50
)

print(
    f"True Positive : {global_true_positive}"
)

print(
    f"True Negative : {global_true_negative}"
)

print(
    f"False Positive: {global_false_positive}"
)

print(
    f"False Negative: {global_false_negative}"
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
    PERFORMANCE_PATH
)

print(
    "\nDone."
)
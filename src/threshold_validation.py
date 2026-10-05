import os
import time
import numpy as np
import pandas as pd

from river import linear_model
from river import preprocessing


# ============================================================
# STEP 12 - CHRONOLOGICAL THRESHOLD VALIDATION
# ============================================================

print("=" * 70)
print("STEP 12 - CHRONOLOGICAL THRESHOLD VALIDATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/processed/ieee_features.csv"

RESULTS_DIR = "results/analysis"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "threshold_walk_forward.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "threshold_walk_forward_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 50000

THRESHOLDS = np.arange(
    0.05,
    0.951,
    0.05
)

MIN_CALIBRATION_TRANSACTIONS = 10000


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
# SORT CHRONOLOGICALLY
# ============================================================

print(
    "\n[2/8] Sorting chronological transaction stream..."
)

data = data.sort_values(
    by=[
        "TransactionDT",
        "TransactionID"
    ]
).reset_index(
    drop=True
)

TOTAL_TRANSACTIONS = len(data)

print(
    "Transactions:",
    TOTAL_TRANSACTIONS
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
# HELPER FUNCTION
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
# THRESHOLD METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
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

    true_positive = int(
        np.sum(
            (y_true == 1)
            &
            (predictions == 1)
        )
    )

    true_negative = int(
        np.sum(
            (y_true == 0)
            &
            (predictions == 0)
        )
    )

    false_positive = int(
        np.sum(
            (y_true == 0)
            &
            (predictions == 1)
        )
    )

    false_negative = int(
        np.sum(
            (y_true == 1)
            &
            (predictions == 0)
        )
    )

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

    return {

        "threshold":
            threshold,

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
            true_positive,

        "true_negative":
            true_negative,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative
    }


# ============================================================
# MODEL
# ============================================================

print(
    "\n[3/8] Creating chronological online model..."
)

model = (
    preprocessing.StandardScaler()
    |
    linear_model.LogisticRegression()
)


# ============================================================
# STORAGE
# ============================================================

all_probabilities = []

all_labels = []

window_results = []


# ============================================================
# STREAM PROCESSING
# ============================================================

print(
    "\n[4/8] Generating chronological predictions..."
)

start_time = time.time()


for index, row in data.iterrows():

    x = build_features(
        row
    )

    y = int(
        row[TARGET]
    )

    # --------------------------------------------------------
    # PREDICT BEFORE LEARNING
    # --------------------------------------------------------

    probabilities = model.predict_proba_one(
        x
    )

    fraud_probability = probabilities.get(
        1,
        0.0
    )

    all_probabilities.append(
        fraud_probability
    )

    all_labels.append(
        y
    )

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

    # ========================================================
    # WINDOW END DETECTION
    # ========================================================

    is_window_end = (
        transaction_number % WINDOW_SIZE == 0
        or
        transaction_number == TOTAL_TRANSACTIONS
    )

    if not is_window_end:
        continue

    # ========================================================
    # CURRENT WINDOW NUMBER
    # ========================================================

    current_window = (
        (transaction_number - 1)
        //
        WINDOW_SIZE
    ) + 1

    # ========================================================
    # STRICT NON-OVERLAPPING WINDOW
    # ========================================================

    window_end = transaction_number

    window_start = (
        (current_window - 1)
        *
        WINDOW_SIZE
    )

    # Python slicing:
    #
    # Window 1:
    #   index 0 ... 49999
    #
    # Window 2:
    #   index 50000 ... 99999
    #
    # Final partial window:
    #   index 550000 ... 590539
    #
    # Therefore there is NO overlap.

    # ========================================================
    # CALIBRATION DATA
    # ========================================================

    calibration_end = window_start

    calibration_start = max(
        0,
        calibration_end - WINDOW_SIZE
    )

    calibration_labels = np.asarray(
        all_labels[
            calibration_start:
            calibration_end
        ]
    )

    calibration_probabilities = np.asarray(
        all_probabilities[
            calibration_start:
            calibration_end
        ]
    )

    # ========================================================
    # THRESHOLD SELECTION
    # ========================================================

    if len(calibration_labels) < MIN_CALIBRATION_TRANSACTIONS:

        selected_threshold = 0.50

        calibration_f1 = np.nan

    else:

        best_f1 = -1.0

        selected_threshold = 0.50

        for threshold in THRESHOLDS:

            result = calculate_metrics(
                calibration_labels,
                calibration_probabilities,
                threshold
            )

            if result["f1"] > best_f1:

                best_f1 = result["f1"]

                selected_threshold = (
                    threshold
                )

        calibration_f1 = best_f1

    # ========================================================
    # VALIDATION WINDOW
    # ========================================================

    window_labels = np.asarray(
        all_labels[
            window_start:
            window_end
        ]
    )

    window_probabilities = np.asarray(
        all_probabilities[
            window_start:
            window_end
        ]
    )

    window_metric = calculate_metrics(
        window_labels,
        window_probabilities,
        selected_threshold
    )

    # ========================================================
    # STORE RESULT
    # ========================================================

    result_row = {

        "window":
            current_window,

        "window_start":
            window_start + 1,

        "window_end":
            window_end,

        "window_transactions":
            window_end - window_start,

        "calibration_start":
            calibration_start + 1
            if calibration_end > calibration_start
            else 0,

        "calibration_end":
            calibration_end,

        "calibration_transactions":
            len(calibration_labels),

        "selected_threshold":
            selected_threshold,

        "calibration_f1":
            calibration_f1,

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
    }

    window_results.append(
        result_row
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"\nWindow {current_window}"
    )

    print(
        "-" * 50
    )

    print(
        "Transactions:",
        window_start + 1,
        "-",
        window_end
    )

    print(
        "Window size:",
        window_end - window_start
    )

    print(
        "Calibration:",
        calibration_start + 1,
        "-",
        calibration_end
    )

    print(
        "Calibration transactions:",
        len(calibration_labels)
    )

    print(
        f"Selected threshold : "
        f"{selected_threshold:.2f}"
    )

    if not np.isnan(
        calibration_f1
    ):

        print(
            f"Calibration F1     : "
            f"{calibration_f1:.4f}"
        )

    print(
        f"Validation F1      : "
        f"{window_metric['f1']:.4f}"
    )

    print(
        f"Validation Recall  : "
        f"{window_metric['recall']:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{window_metric['precision']:.4f}"
    )


# ============================================================
# SAVE WINDOW RESULTS
# ============================================================

print(
    "\n[5/8] Saving walk-forward results..."
)

results = pd.DataFrame(
    window_results
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print(
    "\n[6/8] Calculating validation statistics..."
)

if len(results) > 0:

    summary = pd.DataFrame([{

        "windows":
            len(results),

        "window_size":
            WINDOW_SIZE,

        "total_transactions":
            TOTAL_TRANSACTIONS,

        "mean_threshold":
            results[
                "selected_threshold"
            ].mean(),

        "median_threshold":
            results[
                "selected_threshold"
            ].median(),

        "threshold_std":
            results[
                "selected_threshold"
            ].std(),

        "min_threshold":
            results[
                "selected_threshold"
            ].min(),

        "max_threshold":
            results[
                "selected_threshold"
            ].max(),

        "mean_accuracy":
            results[
                "accuracy"
            ].mean(),

        "std_accuracy":
            results[
                "accuracy"
            ].std(),

        "mean_precision":
            results[
                "precision"
            ].mean(),

        "std_precision":
            results[
                "precision"
            ].std(),

        "mean_recall":
            results[
                "recall"
            ].mean(),

        "std_recall":
            results[
                "recall"
            ].std(),

        "mean_f1":
            results[
                "f1"
            ].mean(),

        "std_f1":
            results[
                "f1"
            ].std(),

        "mean_fpr":
            results[
                "fpr"
            ].mean(),

        "std_fpr":
            results[
                "fpr"
            ].std(),

        "mean_calibration_f1":
            results[
                "calibration_f1"
            ].mean()

    }])

else:

    summary = pd.DataFrame()


summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# THRESHOLD DISTRIBUTION
# ============================================================

print(
    "\nThreshold distribution:"
)

if len(results) > 0:

    print(
        results[
            "selected_threshold"
        ]
        .value_counts()
        .sort_index()
    )


# ============================================================
# WINDOW VALIDATION CHECK
# ============================================================

print(
    "\nWindow boundary validation:"
)

if len(results) > 0:

    overlap_detected = False

    previous_end = 0

    for _, row in results.iterrows():

        current_start = int(
            row["window_start"]
        )

        current_end = int(
            row["window_end"]
        )

        if current_start != previous_end + 1:

            overlap_detected = True

            print(
                "WARNING: Window boundary problem:",
                current_start,
                current_end
            )

        previous_end = current_end

    if not overlap_detected:

        print(
            "OK - All validation windows are chronological "
            "and non-overlapping."
        )

    else:

        print(
            "WARNING - Check window boundaries."
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
    "STEP 12 WALK-FORWARD VALIDATION COMPLETE"
)

print(
    "=" * 70
)


if len(results) > 0:

    print(
        "\nValidation Summary"
    )

    print(
        "-" * 50
    )

    print(
        f"Windows          : "
        f"{len(results)}"
    )

    print(
        f"Total transactions: "
        f"{TOTAL_TRANSACTIONS}"
    )

    print(
        f"Mean threshold   : "
        f"{results['selected_threshold'].mean():.4f}"
    )

    print(
        f"Median threshold : "
        f"{results['selected_threshold'].median():.4f}"
    )

    print(
        f"Threshold std    : "
        f"{results['selected_threshold'].std():.4f}"
    )

    print(
        f"Mean precision   : "
        f"{results['precision'].mean():.4f}"
    )

    print(
        f"Mean recall      : "
        f"{results['recall'].mean():.4f}"
    )

    print(
        f"Mean F1          : "
        f"{results['f1'].mean():.4f}"
    )

    print(
        f"Mean FPR         : "
        f"{results['fpr'].mean():.4f}"
    )


print(
    f"\nProcessing time: "
    f"{elapsed_time:.2f} seconds"
)


print(
    "\nFiles saved:"
)

print(
    OUTPUT_PATH
)

print(
    SUMMARY_PATH
)

print(
    "\nDone."
)
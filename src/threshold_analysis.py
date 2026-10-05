import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# STEP 11 - FRAUD DECISION THRESHOLD ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 11 - FRAUD DECISION THRESHOLD ANALYSIS")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    "data/processed/ieee_features.csv"
)

OUTPUT_DIR = (
    "results/analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TRANSACTIONS = None

THRESHOLDS = np.arange(
    0.05,
    1.00,
    0.05
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n[1/6] Loading feature dataset..."
)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}"
    )


data = pd.read_csv(
    INPUT_FILE
)


print(
    "Dataset shape:",
    data.shape
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print(
    "\n[2/6] Sorting chronological transaction stream..."
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
    "Transactions:",
    len(data)
)


# ============================================================
# CHECK TARGET
# ============================================================

TARGET = "isFraud"


if TARGET not in data.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# LOAD SAVED PREDICTIONS IF AVAILABLE
# ============================================================

PREDICTION_FILE = os.path.join(
    OUTPUT_DIR,
    "threshold_prediction_scores.csv"
)


# ============================================================
# IMPORTANT
# ============================================================
#
# We need probability predictions.
#
# To avoid changing the existing baseline experiment,
# this script trains a separate chronological Logistic
# Regression model and records probability predictions.
#
# ============================================================

print(
    "\n[3/6] Generating chronological probability predictions..."
)


from river import linear_model
from river import preprocessing


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


model = (

    preprocessing.StandardScaler()

    |

    linear_model.LogisticRegression()

)


labels = []

probabilities = []

transaction_indices = []


for index, row in data.iterrows():

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


    y = int(
        row[TARGET]
    )


    # --------------------------------------------------------
    # Predict BEFORE learning
    # --------------------------------------------------------

    probabilities_dict = (
        model.predict_proba_one(x)
    )


    fraud_probability = probabilities_dict.get(
        1,
        0.0
    )


    labels.append(
        y
    )


    probabilities.append(
        fraud_probability
    )


    transaction_indices.append(
        index + 1
    )


    # --------------------------------------------------------
    # Learn AFTER prediction
    # --------------------------------------------------------

    model.learn_one(
        x,
        y
    )


# ============================================================
# SAVE PREDICTION SCORES
# ============================================================

prediction_df = pd.DataFrame({

    "transaction_index":
        transaction_indices,

    "isFraud":
        labels,

    "fraud_probability":
        probabilities

})


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)


print(
    "Probability predictions saved:"
)

print(
    PREDICTION_FILE
)


# ============================================================
# BASELINE AUC METRICS
# ============================================================

print(
    "\n[4/6] Calculating threshold-independent metrics..."
)


y_true = np.asarray(
    labels
)

y_scores = np.asarray(
    probabilities
)


roc_auc = roc_auc_score(
    y_true,
    y_scores
)


pr_auc = average_precision_score(
    y_true,
    y_scores
)


print(
    f"ROC-AUC : {roc_auc:.6f}"
)

print(
    f"PR-AUC  : {pr_auc:.6f}"
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print(
    "\n[5/6] Evaluating decision thresholds..."
)


threshold_results = []


for threshold in THRESHOLDS:

    y_pred = (
        y_scores
        >= threshold
    ).astype(
        int
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ]
    ).ravel()


    if (
        tn + fp
    ) > 0:

        fpr = (
            fp
            /
            (
                tn + fp
            )
        )

    else:

        fpr = 0.0


    accuracy = (
        tn + tp
    ) / len(
        y_true
    )


    threshold_results.append({

        "threshold":
            float(threshold),

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
            int(tp),

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn)

    })


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# SAVE RESULTS
# ============================================================

threshold_file = os.path.join(
    OUTPUT_DIR,
    "threshold_analysis.csv"
)


threshold_df.to_csv(
    threshold_file,
    index=False
)


# ============================================================
# BEST THRESHOLD
# ============================================================

best_index = threshold_df[
    "f1"
].idxmax()


best = threshold_df.loc[
    best_index
]


print(
    "\nThreshold Results"
)

print(
    "-" * 90
)


print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)


print(
    "\n" + "=" * 70
)

print(
    "BEST F1 THRESHOLD"
)

print(
    "=" * 70
)


print(
    f"Threshold    : "
    f"{best['threshold']:.2f}"
)

print(
    f"Accuracy     : "
    f"{best['accuracy']:.6f}"
)

print(
    f"Precision    : "
    f"{best['precision']:.6f}"
)

print(
    f"Recall       : "
    f"{best['recall']:.6f}"
)

print(
    f"F1           : "
    f"{best['f1']:.6f}"
)

print(
    f"FPR          : "
    f"{best['fpr']:.6f}"
)

print(
    f"TP           : "
    f"{int(best['true_positive'])}"
)

print(
    f"TN           : "
    f"{int(best['true_negative'])}"
)

print(
    f"FP           : "
    f"{int(best['false_positive'])}"
)

print(
    f"FN           : "
    f"{int(best['false_negative'])}"
)


# ============================================================
# COMPARISON WITH DEFAULT 0.50
# ============================================================

default_rows = threshold_df[
    np.isclose(
        threshold_df["threshold"],
        0.50
    )
]


if len(default_rows) > 0:

    default = default_rows.iloc[0]


    print(
        "\nComparison with threshold 0.50"
    )

    print(
        "-" * 50
    )

    print(
        f"Default F1 : "
        f"{default['f1']:.6f}"
    )

    print(
        f"Best F1    : "
        f"{best['f1']:.6f}"
    )


    if default["f1"] != 0:

        improvement = (

            (
                best["f1"]
                -
                default["f1"]
            )
            /
            default["f1"]

        ) * 100

    else:

        improvement = 0.0


    print(
        f"F1 change  : "
        f"{improvement:+.2f}%"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "threshold_analysis_summary.csv"
)


summary = pd.DataFrame([{

    "roc_auc":
        roc_auc,

    "pr_auc":
        pr_auc,

    "best_threshold":
        best["threshold"],

    "accuracy":
        best["accuracy"],

    "precision":
        best["precision"],

    "recall":
        best["recall"],

    "f1":
        best["f1"],

    "fpr":
        best["fpr"],

    "true_positive":
        best["true_positive"],

    "true_negative":
        best["true_negative"],

    "false_positive":
        best["false_positive"],

    "false_negative":
        best["false_negative"]

}])


summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print(
    "\nFiles saved:"
)

print(
    threshold_file
)

print(
    summary_file
)

print(
    "\nSTEP 11 COMPLETE"
)

print(
    "=" * 70
)
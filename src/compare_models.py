import os
import pandas as pd


# ============================================================
# STEP 8 - FAIR THREE-MODEL COMPARISON
# ============================================================

print("=" * 70)
print("STEP 8 - FAIR THREE-MODEL COMPARISON")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "results"

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

ANALYSIS_DIR = os.path.join(
    RESULTS_DIR,
    "analysis"
)

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)


# ============================================================
# MODEL RESULT FILES
# ============================================================

FILES = {

    "Static Online LR":
        os.path.join(
            METRICS_DIR,
            "baseline_online_logistic_summary.csv"
        ),

    "ADWIN + Reset LR":
        os.path.join(
            METRICS_DIR,
            "adaptive_adwin_logistic_summary.csv"
        ),

    "ADWIN + Replay LR":
        os.path.join(
            METRICS_DIR,
            "improved_adaptive_logistic_summary.csv"
        )

}


# ============================================================
# LOAD RESULTS
# ============================================================

print("\n[1/5] Loading model results...")

records = []


for model_name, file_path in FILES.items():

    print(
        f"\nLoading: {model_name}"
    )

    print(
        f"File: {file_path}"
    )

    if not os.path.exists(file_path):

        print(
            "WARNING: File not found."
        )

        continue


    df = pd.read_csv(
        file_path
    )


    if df.empty:

        print(
            "WARNING: File is empty."
        )

        continue


    row = df.iloc[0].to_dict()

    row["comparison_model"] = model_name

    records.append(
        row
    )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(records) == 0:

    raise RuntimeError(
        "No model result files were found."
    )


comparison = pd.DataFrame(
    records
)


print(
    f"\nModels loaded: {len(comparison)}"
)


# ============================================================
# CHECK TRANSACTION COUNTS
# ============================================================

print(
    "\n[2/5] Checking experimental consistency..."
)


if "transactions" in comparison.columns:

    transaction_counts = (
        comparison["transactions"]
        .unique()
    )

    print(
        "Transaction counts:",
        transaction_counts
    )


    if len(transaction_counts) == 1:

        print(
            "OK - All models processed "
            "the same number of transactions."
        )

    else:

        print(
            "WARNING - Models processed "
            "different numbers of transactions."
        )


# ============================================================
# SELECT COMPARISON COLUMNS
# ============================================================

metric_columns = [

    "comparison_model",

    "transactions",

    "accuracy",

    "precision",

    "recall",

    "f1",

    "roc_auc",

    "pr_auc",

    "kappa",

    "fpr",

    "drift_events",

    "true_positive",

    "true_negative",

    "false_positive",

    "false_negative",

    "throughput",

    "processing_time_seconds"

]


available_columns = [

    column

    for column in metric_columns

    if column in comparison.columns

]


comparison_table = comparison[
    available_columns
].copy()


# ============================================================
# ROUND NUMERIC VALUES
# ============================================================

numeric_columns = [

    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "kappa",
    "fpr",
    "throughput",
    "processing_time_seconds"

]


for column in numeric_columns:

    if column in comparison_table.columns:

        comparison_table[column] = pd.to_numeric(
            comparison_table[column],
            errors="coerce"
        ).round(6)


# ============================================================
# SAVE COMPLETE COMPARISON
# ============================================================

comparison_path = os.path.join(
    ANALYSIS_DIR,
    "three_model_comparison.csv"
)


comparison_table.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# IEEE TABLE
# ============================================================

ieee_columns = {

    "comparison_model":
        "Model",

    "accuracy":
        "Accuracy",

    "precision":
        "Precision",

    "recall":
        "Recall",

    "f1":
        "F1",

    "roc_auc":
        "ROC-AUC",

    "pr_auc":
        "PR-AUC",

    "kappa":
        "Kappa",

    "fpr":
        "FPR",

    "drift_events":
        "Drifts"

}


ieee_available = [

    column

    for column in ieee_columns

    if column in comparison_table.columns

]


ieee_table = comparison_table[
    ieee_available
].rename(
    columns={
        column: ieee_columns[column]
        for column in ieee_available
    }
)


ieee_path = os.path.join(
    ANALYSIS_DIR,
    "ieee_model_comparison.csv"
)


ieee_table.to_csv(
    ieee_path,
    index=False
)


# ============================================================
# BEST MODEL ANALYSIS
# ============================================================

print(
    "\n[3/5] Determining best-performing models..."
)


if "f1" in comparison.columns:

    best_f1_index = comparison[
        "f1"
    ].idxmax()

    print(
        "\nBest F1:"
    )

    print(
        comparison.loc[
            best_f1_index,
            "comparison_model"
        ],
        f"({comparison.loc[best_f1_index, 'f1']:.6f})"
    )


if "recall" in comparison.columns:

    best_recall_index = comparison[
        "recall"
    ].idxmax()

    print(
        "\nBest Recall:"
    )

    print(
        comparison.loc[
            best_recall_index,
            "comparison_model"
        ],
        f"({comparison.loc[best_recall_index, 'recall']:.6f})"
    )


if "precision" in comparison.columns:

    best_precision_index = comparison[
        "precision"
    ].idxmax()

    print(
        "\nBest Precision:"
    )

    print(
        comparison.loc[
            best_precision_index,
            "comparison_model"
        ],
        f"({comparison.loc[best_precision_index, 'precision']:.6f})"
    )


if "pr_auc" in comparison.columns:

    best_pr_auc_index = comparison[
        "pr_auc"
    ].idxmax()

    print(
        "\nBest PR-AUC:"
    )

    print(
        comparison.loc[
            best_pr_auc_index,
            "comparison_model"
        ],
        f"({comparison.loc[best_pr_auc_index, 'pr_auc']:.6f})"
    )


if "kappa" in comparison.columns:

    best_kappa_index = comparison[
        "kappa"
    ].idxmax()

    print(
        "\nBest Kappa:"
    )

    print(
        comparison.loc[
            best_kappa_index,
            "comparison_model"
        ],
        f"({comparison.loc[best_kappa_index, 'kappa']:.6f})"
    )


# ============================================================
# IMPROVEMENT CALCULATIONS
# ============================================================

print(
    "\n[4/5] Calculating improvements..."
)


baseline_rows = comparison[
    comparison["comparison_model"]
    ==
    "Static Online LR"
]


if len(baseline_rows) > 0:

    baseline = baseline_rows.iloc[0]


    for model_name in [

        "ADWIN + Reset LR",

        "ADWIN + Replay LR"

    ]:

        selected_rows = comparison[
            comparison["comparison_model"]
            ==
            model_name
        ]


        if len(selected_rows) == 0:

            continue


        selected = selected_rows.iloc[0]


        # ----------------------------------------------------
        # F1 CHANGE
        # ----------------------------------------------------

        if baseline["f1"] != 0:

            f1_change = (

                (
                    selected["f1"]
                    -
                    baseline["f1"]
                )
                /
                baseline["f1"]

            ) * 100

        else:

            f1_change = 0.0


        # ----------------------------------------------------
        # RECALL CHANGE
        # ----------------------------------------------------

        if baseline["recall"] != 0:

            recall_change = (

                (
                    selected["recall"]
                    -
                    baseline["recall"]
                )
                /
                baseline["recall"]

            ) * 100

        else:

            recall_change = 0.0


        # ----------------------------------------------------
        # PRECISION CHANGE
        # ----------------------------------------------------

        if baseline["precision"] != 0:

            precision_change = (

                (
                    selected["precision"]
                    -
                    baseline["precision"]
                )
                /
                baseline["precision"]

            ) * 100

        else:

            precision_change = 0.0


        # ----------------------------------------------------
        # PR-AUC CHANGE
        # ----------------------------------------------------

        if baseline["pr_auc"] != 0:

            pr_auc_change = (

                (
                    selected["pr_auc"]
                    -
                    baseline["pr_auc"]
                )
                /
                baseline["pr_auc"]

            ) * 100

        else:

            pr_auc_change = 0.0


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"\n{model_name}"
        )

        print(
            f"F1 change        : "
            f"{f1_change:+.2f}%"
        )

        print(
            f"Recall change    : "
            f"{recall_change:+.2f}%"
        )

        print(
            f"Precision change : "
            f"{precision_change:+.2f}%"
        )

        print(
            f"PR-AUC change    : "
            f"{pr_auc_change:+.2f}%"
        )


else:

    print(
        "WARNING: Static Online LR "
        "baseline was not found."
    )


# ============================================================
# FINAL IEEE TABLE
# ============================================================

print(
    "\n[5/5] IEEE MODEL COMPARISON"
)

print(
    "=" * 110
)


print(
    ieee_table.to_string(
        index=False
    )
)


# ============================================================
# SAVE A TEXT REPORT
# ============================================================

report_path = os.path.join(
    ANALYSIS_DIR,
    "three_model_comparison_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "STEP 8 - THREE MODEL COMPARISON\n"
    )

    file.write(
        "=" * 80
        + "\n\n"
    )

    file.write(
        ieee_table.to_string(
            index=False
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        "Interpretation:\n"
    )

    file.write(
        "- Static Online LR provides the baseline.\n"
    )

    file.write(
        "- ADWIN + Reset improves recall and F1 "
        "but increases false positives.\n"
    )

    file.write(
        "- ADWIN + Replay reduces drift events "
        "but does not improve final F1 over baseline.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\nFiles saved:"
)

print(
    comparison_path
)

print(
    ieee_path
)

print(
    report_path
)


print(
    "\nSTEP 8 COMPLETE"
)

print(
    "=" * 70
)
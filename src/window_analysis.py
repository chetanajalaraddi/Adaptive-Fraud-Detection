import os
import pandas as pd
import numpy as np


# ============================================================
# STEP 9 - WINDOW-BASED PERFORMANCE ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 9 - CHRONOLOGICAL WINDOW-BASED ANALYSIS")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

METRICS_DIR = "results/metrics"
ANALYSIS_DIR = "results/analysis"

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

FILES = {

    "Static Online LR":
        "baseline_online_logistic.csv",

    "ADWIN + Reset LR":
        "adaptive_adwin_logistic.csv",

    "ADWIN + Replay LR":
        "improved_adaptive_logistic.csv"

}


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 10000


# ============================================================
# LOAD MODEL PERFORMANCE
# ============================================================

print("\n[1/5] Loading performance histories...")

all_results = []


for model_name, filename in FILES.items():

    path = os.path.join(
        METRICS_DIR,
        filename
    )


    if not os.path.exists(path):

        print(
            f"WARNING: Missing {path}"
        )

        continue


    df = pd.read_csv(
        path
    )


    if df.empty:

        print(
            f"WARNING: Empty file {path}"
        )

        continue


    print(
        f"{model_name}: {len(df)} records"
    )


    df["model"] = model_name

    all_results.append(
        df
    )


if len(all_results) == 0:

    raise RuntimeError(
        "No performance files found."
    )


# ============================================================
# COMBINE
# ============================================================

performance = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# CREATE WINDOW NUMBER
# ============================================================

print(
    "\n[2/5] Creating chronological windows..."
)


performance["window"] = (
    (
        performance[
            "transactions_processed"
        ]
        -
        1
    )
    //
    WINDOW_SIZE
) + 1


# ============================================================
# SAVE RAW WINDOW DATA
# ============================================================

raw_path = os.path.join(
    ANALYSIS_DIR,
    "window_performance_all_models.csv"
)


performance.to_csv(
    raw_path,
    index=False
)


# ============================================================
# METRICS
# ============================================================

metrics_to_analyze = [

    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "kappa"

]


available_metrics = [

    metric

    for metric in metrics_to_analyze

    if metric in performance.columns

]


# ============================================================
# WINDOW SUMMARY
# ============================================================

print(
    "\n[3/5] Generating window summaries..."
)


window_summary = (
    performance
    .groupby(
        [
            "model",
            "window"
        ]
    )[available_metrics]
    .mean()
    .reset_index()
)


# ============================================================
# ADD TRANSACTION RANGE
# ============================================================

window_summary[
    "window_start"
] = (
    (
        window_summary["window"]
        -
        1
    )
    *
    WINDOW_SIZE
) + 1


window_summary[
    "window_end"
] = (
    window_summary["window"]
    *
    WINDOW_SIZE
)


# ============================================================
# SAVE WINDOW SUMMARY
# ============================================================

window_path = os.path.join(
    ANALYSIS_DIR,
    "window_performance_summary.csv"
)


window_summary.to_csv(
    window_path,
    index=False
)


# ============================================================
# MEAN AND STANDARD DEVIATION
# ============================================================

print(
    "\n[4/5] Calculating window statistics..."
)


statistics = []


for model_name in performance["model"].unique():

    model_data = performance[
        performance["model"]
        ==
        model_name
    ]


    record = {

        "model":
            model_name,

        "windows":
            len(model_data)

    }


    for metric in available_metrics:

        values = pd.to_numeric(
            model_data[metric],
            errors="coerce"
        ).dropna()


        if len(values) > 0:

            record[
                f"{metric}_mean"
            ] = values.mean()


            record[
                f"{metric}_std"
            ] = values.std()


            record[
                f"{metric}_min"
            ] = values.min()


            record[
                f"{metric}_max"
            ] = values.max()

        else:

            record[
                f"{metric}_mean"
            ] = np.nan


            record[
                f"{metric}_std"
            ] = np.nan


            record[
                f"{metric}_min"
            ] = np.nan


            record[
                f"{metric}_max"
            ] = np.nan


    statistics.append(
        record
    )


statistics_df = pd.DataFrame(
    statistics
)


statistics_path = os.path.join(
    ANALYSIS_DIR,
    "window_statistics.csv"
)


statistics_df.to_csv(
    statistics_path,
    index=False
)


# ============================================================
# PRINT STATISTICS
# ============================================================

print(
    "\nWindow Statistics"
)

print(
    "-" * 100
)


display_columns = [

    "model",
    "windows",

    "f1_mean",
    "f1_std",

    "recall_mean",
    "recall_std",

    "precision_mean",
    "precision_std",

    "pr_auc_mean",
    "pr_auc_std"

]


available_display = [

    column

    for column in display_columns

    if column in statistics_df.columns

]


print(
    statistics_df[
        available_display
    ].to_string(
        index=False
    )
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print(
    "\n[5/5] Final window analysis complete"
)

print(
    "=" * 70
)

print(
    "Saved:"
)

print(
    raw_path
)

print(
    window_path
)

print(
    statistics_path
)

print(
    "\nSTEP 9 COMPLETE"
)

print(
    "=" * 70
)
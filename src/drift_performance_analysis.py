import os
import pandas as pd
import numpy as np


# ============================================================
# STEP 10 - DRIFT VS PERFORMANCE ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 10 - DRIFT VS PERFORMANCE ANALYSIS")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

DRIFT_DIR = "results/drift"
ANALYSIS_DIR = "results/analysis"

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

RESET_DRIFT_FILE = os.path.join(
    DRIFT_DIR,
    "adwin_drift_events.csv"
)

REPLAY_DRIFT_FILE = os.path.join(
    DRIFT_DIR,
    "improved_adaptive_drift_events.csv"
)

WINDOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "window_performance_all_models.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/7] Loading drift and performance data...")


if not os.path.exists(RESET_DRIFT_FILE):

    raise FileNotFoundError(
        f"Missing file:\n{RESET_DRIFT_FILE}"
    )


if not os.path.exists(REPLAY_DRIFT_FILE):

    raise FileNotFoundError(
        f"Missing file:\n{REPLAY_DRIFT_FILE}"
    )


if not os.path.exists(WINDOW_FILE):

    raise FileNotFoundError(
        f"Missing file:\n{WINDOW_FILE}\n"
        "Run window_analysis.py first."
    )


reset_drift = pd.read_csv(
    RESET_DRIFT_FILE
)

replay_drift = pd.read_csv(
    REPLAY_DRIFT_FILE
)

performance = pd.read_csv(
    WINDOW_FILE
)


print(
    f"ADWIN Reset drift events  : {len(reset_drift)}"
)

print(
    f"ADWIN Replay drift events : {len(replay_drift)}"
)

print(
    f"Performance records       : {len(performance)}"
)


# ============================================================
# DRIFT SPACING ANALYSIS
# ============================================================

print(
    "\n[2/7] Analyzing drift spacing..."
)


def calculate_drift_spacing(
    drift_df
):

    df = drift_df.copy()

    df = df.sort_values(
        "transaction_index"
    ).reset_index(
        drop=True
    )


    if len(df) < 2:

        df["distance_from_previous_drift"] = np.nan

        return df


    df[
        "distance_from_previous_drift"
    ] = (
        df["transaction_index"]
        .diff()
    )


    return df


reset_drift = calculate_drift_spacing(
    reset_drift
)

replay_drift = calculate_drift_spacing(
    replay_drift
)


# ============================================================
# DRIFT SPACING SUMMARY
# ============================================================

reset_spacing = reset_drift[
    "distance_from_previous_drift"
].dropna()

replay_spacing = replay_drift[
    "distance_from_previous_drift"
].dropna()


print(
    "\nADWIN + Reset"
)

if len(reset_spacing) > 0:

    print(
        f"Mean distance   : "
        f"{reset_spacing.mean():.2f}"
    )

    print(
        f"Median distance : "
        f"{reset_spacing.median():.2f}"
    )

    print(
        f"Minimum distance: "
        f"{reset_spacing.min():.2f}"
    )

    print(
        f"Maximum distance: "
        f"{reset_spacing.max():.2f}"
    )


print(
    "\nADWIN + Replay"
)

if len(replay_spacing) > 0:

    print(
        f"Mean distance   : "
        f"{replay_spacing.mean():.2f}"
    )

    print(
        f"Median distance : "
        f"{replay_spacing.median():.2f}"
    )

    print(
        f"Minimum distance: "
        f"{replay_spacing.min():.2f}"
    )

    print(
        f"Maximum distance: "
        f"{replay_spacing.max():.2f}"
    )


# ============================================================
# MAP DRIFT TO CHRONOLOGICAL WINDOW
# ============================================================

print(
    "\n[3/7] Mapping drift events to windows..."
)


WINDOW_SIZE = 10000


reset_drift[
    "window"
] = (
    (
        reset_drift[
            "transaction_index"
        ]
        -
        1
    )
    //
    WINDOW_SIZE
) + 1


replay_drift[
    "window"
] = (
    (
        replay_drift[
            "transaction_index"
        ]
        -
        1
    )
    //
    WINDOW_SIZE
) + 1


# ============================================================
# DRIFTS PER WINDOW
# ============================================================

reset_counts = (
    reset_drift
    .groupby("window")
    .size()
    .reset_index(
        name="reset_drift_count"
    )
)


replay_counts = (
    replay_drift
    .groupby("window")
    .size()
    .reset_index(
        name="replay_drift_count"
    )
)


# ============================================================
# MERGE WITH PERFORMANCE
# ============================================================

print(
    "\n[4/7] Combining drift frequency with model performance..."
)


window_data = performance.copy()


window_data = window_data.merge(
    reset_counts,
    on="window",
    how="left"
)


window_data = window_data.merge(
    replay_counts,
    on="window",
    how="left"
)


window_data[
    "reset_drift_count"
] = window_data[
    "reset_drift_count"
].fillna(0)


window_data[
    "replay_drift_count"
] = window_data[
    "replay_drift_count"
].fillna(0)


# ============================================================
# SAVE DRIFT-WINDOW DATA
# ============================================================

drift_window_path = os.path.join(
    ANALYSIS_DIR,
    "drift_performance_by_window.csv"
)


window_data.to_csv(
    drift_window_path,
    index=False
)


# ============================================================
# MODEL-SPECIFIC CORRELATION
# ============================================================

print(
    "\n[5/7] Calculating drift-performance correlations..."
)


correlation_records = []


for model_name in window_data[
    "model"
].unique():

    model_data = window_data[
        window_data["model"]
        ==
        model_name
    ].copy()


    if (
        "f1" not in model_data.columns
        or
        "recall" not in model_data.columns
    ):

        continue


    # --------------------------------------------------------
    # Select appropriate drift count
    # --------------------------------------------------------

    if model_name == "ADWIN + Reset LR":

        drift_column = (
            "reset_drift_count"
        )

    elif model_name == "ADWIN + Replay LR":

        drift_column = (
            "replay_drift_count"
        )

    else:

        drift_column = None


    if drift_column is None:

        continue


    valid = model_data[
        [
            drift_column,
            "f1",
            "recall",
            "precision",
            "accuracy",
            "roc_auc"
        ]
    ].dropna()


    if len(valid) < 3:

        continue


    record = {

        "model":
            model_name,

        "windows":
            len(valid),

        "drift_f1_correlation":
            valid[
                drift_column
            ].corr(
                valid["f1"]
            ),

        "drift_recall_correlation":
            valid[
                drift_column
            ].corr(
                valid["recall"]
            ),

        "drift_precision_correlation":
            valid[
                drift_column
            ].corr(
                valid["precision"]
            ),

        "drift_accuracy_correlation":
            valid[
                drift_column
            ].corr(
                valid["accuracy"]
            ),

        "drift_roc_auc_correlation":
            valid[
                drift_column
            ].corr(
                valid["roc_auc"]
            )

    }


    correlation_records.append(
        record
    )


correlation_df = pd.DataFrame(
    correlation_records
)


# ============================================================
# SAVE CORRELATIONS
# ============================================================

correlation_path = os.path.join(
    ANALYSIS_DIR,
    "drift_performance_correlations.csv"
)


correlation_df.to_csv(
    correlation_path,
    index=False
)


# ============================================================
# PRINT CORRELATIONS
# ============================================================

print(
    "\nDrift-Performance Correlations"
)

print(
    "-" * 100
)


if len(correlation_df) > 0:

    print(
        correlation_df.to_string(
            index=False
        )
    )

else:

    print(
        "No valid correlation results."
    )


# ============================================================
# DRIFT EVENT PERFORMANCE
# ============================================================

print(
    "\n[6/7] Analyzing performance at detected drift points..."
)


event_records = []


for _, row in reset_drift.iterrows():

    event_records.append({

        "detector":
            "ADWIN + Reset",

        "transaction_index":
            row["transaction_index"],

        "drift_number":
            row["drift_number"],

        "accuracy":
            row["accuracy_at_detection"],

        "precision":
            row["precision_at_detection"],

        "recall":
            row["recall_at_detection"],

        "f1":
            row["f1_at_detection"],

        "roc_auc":
            row["roc_auc_at_detection"],

        "adwin_width":
            row["adwin_width"]

    })


for _, row in replay_drift.iterrows():

    event_records.append({

        "detector":
            "ADWIN + Replay",

        "transaction_index":
            row["transaction_index"],

        "drift_number":
            row["drift_number"],

        "accuracy":
            row["accuracy_at_detection"],

        "precision":
            row["precision_at_detection"],

        "recall":
            row["recall_at_detection"],

        "f1":
            row["f1_at_detection"],

        "roc_auc":
            row["roc_auc_at_detection"],

        "adwin_width":
            row["adwin_width"]

    })


event_performance = pd.DataFrame(
    event_records
)


event_path = os.path.join(
    ANALYSIS_DIR,
    "drift_event_performance.csv"
)


event_performance.to_csv(
    event_path,
    index=False
)


# ============================================================
# EVENT PERFORMANCE SUMMARY
# ============================================================

event_summary = (
    event_performance
    .groupby("detector")
    [
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "adwin_width"
        ]
    ]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max"
        ]
    )
)


event_summary_path = os.path.join(
    ANALYSIS_DIR,
    "drift_event_performance_summary.csv"
)


event_summary.to_csv(
    event_summary_path
)


print(
    "\nPerformance at Drift Detection"
)

print(
    "-" * 100
)


print(
    event_summary.to_string()
)


# ============================================================
# CONCENTRATED DRIFT WINDOWS
# ============================================================

print(
    "\n[7/7] Identifying high-drift windows..."
)


reset_high_drift = (
    reset_drift
    .groupby("window")
    .size()
    .sort_values(
        ascending=False
    )
)


replay_high_drift = (
    replay_drift
    .groupby("window")
    .size()
    .sort_values(
        ascending=False
    )
)


print(
    "\nTop ADWIN Reset drift windows:"
)

print(
    reset_high_drift.head(10).to_string()
)


print(
    "\nTop ADWIN Replay drift windows:"
)

print(
    replay_high_drift.head(10).to_string()
)


# ============================================================
# SAVE HIGH DRIFT WINDOWS
# ============================================================

high_drift_records = []


for window, count in reset_high_drift.items():

    high_drift_records.append({

        "detector":
            "ADWIN + Reset",

        "window":
            window,

        "drift_count":
            count

    })


for window, count in replay_high_drift.items():

    high_drift_records.append({

        "detector":
            "ADWIN + Replay",

        "window":
            window,

        "drift_count":
            count

    })


high_drift_df = pd.DataFrame(
    high_drift_records
)


high_drift_path = os.path.join(
    ANALYSIS_DIR,
    "high_drift_windows.csv"
)


high_drift_df.to_csv(
    high_drift_path,
    index=False
)


# ============================================================
# FINAL SUMMARY REPORT
# ============================================================

report_path = os.path.join(
    ANALYSIS_DIR,
    "drift_performance_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "STEP 10 - DRIFT VS PERFORMANCE ANALYSIS\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"ADWIN Reset drift events: "
        f"{len(reset_drift)}\n"
    )

    file.write(
        f"ADWIN Replay drift events: "
        f"{len(replay_drift)}\n\n"
    )


    if len(reset_spacing) > 0:

        file.write(
            "ADWIN Reset drift spacing\n"
        )

        file.write(
            f"Mean: {reset_spacing.mean():.2f}\n"
        )

        file.write(
            f"Median: {reset_spacing.median():.2f}\n\n"
        )


    if len(replay_spacing) > 0:

        file.write(
            "ADWIN Replay drift spacing\n"
        )

        file.write(
            f"Mean: {replay_spacing.mean():.2f}\n"
        )

        file.write(
            f"Median: {replay_spacing.median():.2f}\n\n"
        )


    file.write(
        "Correlation analysis\n"
    )

    file.write(
        "-" * 70
        + "\n"
    )

    if len(correlation_df) > 0:

        file.write(
            correlation_df.to_string(
                index=False
            )
        )

    else:

        file.write(
            "No correlation results available."
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\nFiles saved:"
)

print(
    drift_window_path
)

print(
    correlation_path
)

print(
    event_path
)

print(
    event_summary_path
)

print(
    high_drift_path
)

print(
    report_path
)


print(
    "\nSTEP 10 COMPLETE"
)

print(
    "=" * 70
)
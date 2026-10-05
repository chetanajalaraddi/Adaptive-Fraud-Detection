import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL THREE-DATASET ADAPTIVE FRAUD DETECTION ANALYSIS
# ============================================================

print("=" * 90)
print("FINAL THREE-DATASET ADAPTIVE FRAUD DETECTION ANALYSIS")
print("=" * 90)


# ============================================================
# PATHS
# ============================================================

ANALYSIS_DIR = "results/analysis"
FIGURES_DIR = "results/figures/three_dataset"

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "three_dataset_model_comparison.csv"
)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Required file not found:\n{INPUT_FILE}"
    )


data = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# BASIC CHECK
# ============================================================

required_columns = [
    "dataset",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "kappa",
    "fpr",
    "drift_events",
    "transactions",
    "throughput"
]


missing = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


print("\nLoaded results:")
print(
    data.to_string(
        index=False
    )
)


# ============================================================
# DATASET ORDER
# ============================================================

DATASET_ORDER = [
    "IEEE-CIS",
    "PaySim",
    "Credit Card"
]


MODEL_ORDER = [
    "Baseline",
    "ADWIN Reset",
    "ADWIN Replay"
]


data["dataset"] = pd.Categorical(
    data["dataset"],
    categories=DATASET_ORDER,
    ordered=True
)

data["model"] = pd.Categorical(
    data["model"],
    categories=MODEL_ORDER,
    ordered=True
)

data = data.sort_values(
    ["dataset", "model"]
).reset_index(
    drop=True
)


# ============================================================
# 1. MODEL AVERAGES
# ============================================================

print("\n")
print("=" * 90)
print("1. MODEL AVERAGES ACROSS THREE DATASETS")
print("=" * 90)


model_average = (
    data
    .groupby(
        "model",
        observed=False
    )[
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "kappa",
            "fpr",
            "drift_events",
            "throughput"
        ]
    ]
    .mean()
    .reset_index()
)


print(
    model_average.to_string(
        index=False
    )
)


model_average.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "final_three_dataset_model_averages.csv"
    ),
    index=False
)


# ============================================================
# 2. BEST MODEL PER DATASET
# ============================================================

print("\n")
print("=" * 90)
print("2. BEST MODEL PER DATASET")
print("=" * 90)


best_rows = []


for dataset_name in DATASET_ORDER:

    subset = data[
        data["dataset"] == dataset_name
    ]

    if subset.empty:
        continue

    best = subset.loc[
        subset["f1"].idxmax()
    ]

    best_rows.append({

        "dataset":
            dataset_name,

        "best_model":
            str(best["model"]),

        "accuracy":
            best["accuracy"],

        "precision":
            best["precision"],

        "recall":
            best["recall"],

        "f1":
            best["f1"],

        "kappa":
            best["kappa"],

        "fpr":
            best["fpr"],

        "drift_events":
            best["drift_events"],

        "throughput":
            best["throughput"]
    })

    print(
        f"{dataset_name}: "
        f"{best['model']} "
        f"(F1 = {best['f1']:.4f})"
    )


best_models = pd.DataFrame(
    best_rows
)


best_models.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "final_best_model_per_dataset.csv"
    ),
    index=False
)


# ============================================================
# 3. REPLAY VS RESET
# ============================================================

print("\n")
print("=" * 90)
print("3. ADWIN REPLAY VS ADWIN RESET")
print("=" * 90)


replay_rows = []


for dataset_name in DATASET_ORDER:

    subset = data[
        data["dataset"] == dataset_name
    ]


    reset_rows = subset[
        subset["model"] == "ADWIN Reset"
    ]

    replay_model_rows = subset[
        subset["model"] == "ADWIN Replay"
    ]


    if reset_rows.empty or replay_model_rows.empty:

        continue


    reset = reset_rows.iloc[0]

    replay = replay_model_rows.iloc[0]


    f1_change = (
        replay["f1"]
        -
        reset["f1"]
    )


    precision_change = (
        replay["precision"]
        -
        reset["precision"]
    )


    recall_change = (
        replay["recall"]
        -
        reset["recall"]
    )


    drift_change = (
        replay["drift_events"]
        -
        reset["drift_events"]
    )


    replay_rows.append({

        "dataset":
            dataset_name,

        "reset_f1":
            reset["f1"],

        "replay_f1":
            replay["f1"],

        "f1_change":
            f1_change,

        "reset_precision":
            reset["precision"],

        "replay_precision":
            replay["precision"],

        "precision_change":
            precision_change,

        "reset_recall":
            reset["recall"],

        "replay_recall":
            replay["recall"],

        "recall_change":
            recall_change,

        "reset_drift_events":
            reset["drift_events"],

        "replay_drift_events":
            replay["drift_events"],

        "drift_event_change":
            drift_change
    })


    print(
        f"\n{dataset_name}"
    )

    print(
        f"F1 change: "
        f"{f1_change:+.4f}"
    )

    print(
        f"Precision change: "
        f"{precision_change:+.4f}"
    )

    print(
        f"Recall change: "
        f"{recall_change:+.4f}"
    )

    print(
        f"Drift-event change: "
        f"{drift_change:+.0f}"
    )


replay_vs_reset = pd.DataFrame(
    replay_rows
)


replay_vs_reset.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "final_replay_vs_reset_analysis.csv"
    ),
    index=False
)


# ============================================================
# 4. F1 COMPARISON TABLE
# ============================================================

print("\n")
print("=" * 90)
print("4. F1 COMPARISON")
print("=" * 90)


f1_table = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="f1"
    )
    .reindex(DATASET_ORDER)
)


print(
    f1_table.to_string()
)


f1_table.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "final_f1_comparison.csv"
    )
)


# ============================================================
# 5. RELATIVE F1 IMPROVEMENT
# ============================================================

print("\n")
print("=" * 90)
print("5. RELATIVE F1 CHANGE AGAINST BASELINE")
print("=" * 90)


relative_rows = []


for dataset_name in DATASET_ORDER:

    subset = data[
        data["dataset"] == dataset_name
    ]


    baseline_rows = subset[
        subset["model"] == "Baseline"
    ]

    reset_rows = subset[
        subset["model"] == "ADWIN Reset"
    ]

    replay_rows = subset[
        subset["model"] == "ADWIN Replay"
    ]


    if (
        baseline_rows.empty
        or reset_rows.empty
        or replay_rows.empty
    ):
        continue


    baseline = baseline_rows.iloc[0]

    reset = reset_rows.iloc[0]

    replay = replay_rows.iloc[0]


    baseline_f1 = baseline["f1"]


    reset_change = (
        (
            reset["f1"]
            -
            baseline_f1
        )
        /
        baseline_f1
        *
        100
    )


    replay_change = (
        (
            replay["f1"]
            -
            baseline_f1
        )
        /
        baseline_f1
        *
        100
    )


    relative_rows.append({

        "dataset":
            dataset_name,

        "baseline_f1":
            baseline_f1,

        "reset_f1":
            reset["f1"],

        "replay_f1":
            replay["f1"],

        "reset_relative_change_percent":
            reset_change,

        "replay_relative_change_percent":
            replay_change
    })


    print(
        f"{dataset_name}:"
    )

    print(
        f"  Reset: "
        f"{reset_change:+.2f}%"
    )

    print(
        f"  Replay: "
        f"{replay_change:+.2f}%"
    )


relative_analysis = pd.DataFrame(
    relative_rows
)


relative_analysis.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "final_relative_f1_analysis.csv"
    ),
    index=False
)


# ============================================================
# 6. DRIFT EVENTS
# ============================================================

print("\n")
print("=" * 90)
print("6. DRIFT EVENT ANALYSIS")
print("=" * 90)


drift_table = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="drift_events"
    )
    .reindex(DATASET_ORDER)
)


print(
    drift_table.to_string()
)


drift_table.to_csv(
    os.path.join(
        ANALYSIS_DIR,
        "final_drift_event_comparison.csv"
    )
)


# ============================================================
# 7. PERFORMANCE VS ADAPTATION
# ============================================================

print("\n")
print("=" * 90)
print("7. PERFORMANCE VS ADAPTATION")
print("=" * 90)


for dataset_name in DATASET_ORDER:

    subset = data[
        data["dataset"] == dataset_name
    ]

    print(
        f"\n{dataset_name}"
    )

    for _, row in subset.iterrows():

        print(
            f"{row['model']}: "
            f"F1={row['f1']:.4f}, "
            f"Drifts={row['drift_events']:.0f}, "
            f"FPR={row['fpr']:.6f}"
        )


# ============================================================
# 8. FIGURE - F1 SCORE
# ============================================================

print("\n")
print("Generating F1 comparison figure...")


f1_plot = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="f1"
    )
    .reindex(DATASET_ORDER)
)


ax = f1_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "F1 Score Comparison Across Fraud Detection Datasets"
)

ax.set_xlabel(
    "Dataset"
)

ax.set_ylabel(
    "F1 Score"
)

ax.set_xticklabels(
    DATASET_ORDER,
    rotation=0
)

ax.legend(
    title="Model"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "f1_comparison_three_datasets.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. FIGURE - PRECISION
# ============================================================

print("Generating precision comparison figure...")


precision_plot = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="precision"
    )
    .reindex(DATASET_ORDER)
)


ax = precision_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Precision Comparison Across Fraud Detection Datasets"
)

ax.set_xlabel(
    "Dataset"
)

ax.set_ylabel(
    "Precision"
)

ax.set_xticklabels(
    DATASET_ORDER,
    rotation=0
)

ax.legend(
    title="Model"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "precision_comparison_three_datasets.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. FIGURE - RECALL
# ============================================================

print("Generating recall comparison figure...")


recall_plot = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="recall"
    )
    .reindex(DATASET_ORDER)
)


ax = recall_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Recall Comparison Across Fraud Detection Datasets"
)

ax.set_xlabel(
    "Dataset"
)

ax.set_ylabel(
    "Recall"
)

ax.set_xticklabels(
    DATASET_ORDER,
    rotation=0
)

ax.legend(
    title="Model"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "recall_comparison_three_datasets.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 11. FIGURE - DRIFT EVENTS
# ============================================================

print("Generating drift-event comparison figure...")


drift_plot = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="drift_events"
    )
    .reindex(DATASET_ORDER)
)


ax = drift_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Detected Concept Drift Events Across Datasets"
)

ax.set_xlabel(
    "Dataset"
)

ax.set_ylabel(
    "Number of Drift Events"
)

ax.set_xticklabels(
    DATASET_ORDER,
    rotation=0
)

ax.legend(
    title="Model"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "drift_events_three_datasets.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 12. FIGURE - FPR
# ============================================================

print("Generating FPR comparison figure...")


fpr_plot = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="fpr"
    )
    .reindex(DATASET_ORDER)
)


ax = fpr_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "False Positive Rate Across Fraud Detection Datasets"
)

ax.set_xlabel(
    "Dataset"
)

ax.set_ylabel(
    "False Positive Rate"
)

ax.set_xticklabels(
    DATASET_ORDER,
    rotation=0
)

ax.legend(
    title="Model"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "fpr_comparison_three_datasets.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 13. FIGURE - THROUGHPUT
# ============================================================

print("Generating throughput comparison figure...")


throughput_plot = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="throughput"
    )
    .reindex(DATASET_ORDER)
)


ax = throughput_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Online Processing Throughput Across Datasets"
)

ax.set_xlabel(
    "Dataset"
)

ax.set_ylabel(
    "Transactions per Second"
)

ax.set_xticklabels(
    DATASET_ORDER,
    rotation=0
)

ax.legend(
    title="Model"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "throughput_three_datasets.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 14. FINAL EXPERIMENT SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("FINAL EXPERIMENT SUMMARY")
print("=" * 90)


summary_rows = []


for dataset_name in DATASET_ORDER:

    subset = data[
        data["dataset"] == dataset_name
    ]

    if subset.empty:
        continue


    baseline = subset[
        subset["model"] == "Baseline"
    ].iloc[0]

    reset = subset[
        subset["model"] == "ADWIN Reset"
    ].iloc[0]

    replay = subset[
        subset["model"] == "ADWIN Replay"
    ].iloc[0]


    best = subset.loc[
        subset["f1"].idxmax()
    ]


    summary_rows.append({

        "dataset":
            dataset_name,

        "baseline_f1":
            baseline["f1"],

        "reset_f1":
            reset["f1"],

        "replay_f1":
            replay["f1"],

        "best_model":
            str(best["model"]),

        "best_f1":
            best["f1"],

        "baseline_drift_events":
            baseline["drift_events"],

        "reset_drift_events":
            reset["drift_events"],

        "replay_drift_events":
            replay["drift_events"],

        "replay_vs_reset_f1_change":
            replay["f1"] - reset["f1"],

        "replay_vs_reset_drift_change":
            replay["drift_events"]
            -
            reset["drift_events"]
    })


final_summary = pd.DataFrame(
    summary_rows
)


print(
    final_summary.to_string(
        index=False
    )
)


FINAL_SUMMARY_PATH = os.path.join(
    ANALYSIS_DIR,
    "final_three_dataset_experiment_summary.csv"
)


final_summary.to_csv(
    FINAL_SUMMARY_PATH,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 90)
print("FILES GENERATED")
print("=" * 90)

print(
    f"Analysis directory: "
    f"{ANALYSIS_DIR}"
)

print(
    f"Figures directory: "
    f"{FIGURES_DIR}"
)

print("\nFinal three-dataset analysis completed successfully.")
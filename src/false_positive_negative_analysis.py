import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# ============================================================

print("=" * 90)
print("FALSE POSITIVE AND FALSE NEGATIVE ANALYSIS")
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
# INPUT FILE
# ============================================================

INPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "three_dataset_model_comparison.csv"
)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"\nRequired file not found:\n{INPUT_FILE}\n\n"
        "Run your three-dataset experiment first."
    )


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(INPUT_FILE)


print("\nLoaded file:")
print(INPUT_FILE)

print("\nAvailable columns:")
print(
    data.columns.tolist()
)


# ============================================================
# CHECK FOR CONFUSION-MATRIX COLUMNS
# ============================================================

required_columns = [
    "dataset",
    "model",
    "tp",
    "tn",
    "fp",
    "fn"
]


missing = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing:

    print("\n" + "=" * 90)
    print("ERROR: FP/FN DATA NOT AVAILABLE")
    print("=" * 90)

    print(
        "\nThe following columns are missing:"
    )

    for column in missing:
        print(
            f"  - {column}"
        )

    print(
        "\nYour current CSV contains performance metrics "
        "such as F1, precision, recall and FPR, but it does "
        "not contain the actual FP/FN counts."
    )

    print(
        "\nTherefore, exact False Positive and False Negative "
        "transaction counts cannot be recovered safely from "
        "the current CSV."
    )

    print(
        "\nYou need to modify the experiment code that generates:"
    )

    print(
        "results/analysis/three_dataset_model_comparison.csv"
    )

    print(
        "\nto save these four values:"
    )

    print(
        "TP, TN, FP, FN"
    )

    raise SystemExit


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

for column in [
    "tp",
    "tn",
    "fp",
    "fn"
]:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

if data[
    [
        "tp",
        "tn",
        "fp",
        "fn"
    ]
].isnull().any().any():

    raise ValueError(
        "\nTP/TN/FP/FN contains invalid or missing values."
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
    [
        "dataset",
        "model"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 1. FP / FN TABLE
# ============================================================

print("\n")
print("=" * 90)
print("FALSE POSITIVE / FALSE NEGATIVE RESULTS")
print("=" * 90)


fp_fn_results = data[
    [
        "dataset",
        "model",
        "fp",
        "fn"
    ]
].copy()


print(
    "\n"
    + fp_fn_results.to_string(
        index=False
    )
)


# ============================================================
# 2. SAVE FP/FN CSV
# ============================================================

OUTPUT_CSV = os.path.join(
    ANALYSIS_DIR,
    "false_positive_negative_results.csv"
)


fp_fn_results.to_csv(
    OUTPUT_CSV,
    index=False
)


print(
    f"\nFP/FN CSV saved to:\n{OUTPUT_CSV}"
)


# ============================================================
# 3. TOTAL FP AND FN
# ============================================================

total_fp = int(
    data["fp"].sum()
)


total_fn = int(
    data["fn"].sum()
)


print("\n")
print("=" * 90)
print("TOTAL FALSE POSITIVE / FALSE NEGATIVE")
print("=" * 90)


print(
    f"Total False Positives: {total_fp}"
)

print(
    f"Total False Negatives: {total_fn}"
)


# ============================================================
# 4. DATASET-WISE FP/FN
# ============================================================

print("\n")
print("=" * 90)
print("DATASET-WISE FALSE POSITIVE / FALSE NEGATIVE")
print("=" * 90)


dataset_summary = (
    data
    .groupby(
        "dataset",
        observed=False
    )[
        [
            "fp",
            "fn"
        ]
    ]
    .sum()
    .reset_index()
)


print(
    dataset_summary.to_string(
        index=False
    )
)


DATASET_SUMMARY_FILE = os.path.join(
    ANALYSIS_DIR,
    "dataset_false_positive_negative.csv"
)


dataset_summary.to_csv(
    DATASET_SUMMARY_FILE,
    index=False
)


# ============================================================
# 5. MODEL-WISE FP/FN
# ============================================================

print("\n")
print("=" * 90)
print("MODEL-WISE FALSE POSITIVE / FALSE NEGATIVE")
print("=" * 90)


model_summary = (
    data
    .groupby(
        "model",
        observed=False
    )[
        [
            "fp",
            "fn"
        ]
    ]
    .sum()
    .reset_index()
)


print(
    model_summary.to_string(
        index=False
    )
)


MODEL_SUMMARY_FILE = os.path.join(
    ANALYSIS_DIR,
    "model_false_positive_negative.csv"
)


model_summary.to_csv(
    MODEL_SUMMARY_FILE,
    index=False
)


# ============================================================
# 6. FP GRAPH
# ============================================================

print("\nGenerating False Positive graph...")


fp_plot = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="fp"
    )
    .reindex(DATASET_ORDER)
)


ax = fp_plot.plot(
    kind="bar",
    figsize=(10, 6)
)


ax.set_title(
    "False Positive Transactions"
)


ax.set_xlabel(
    "Dataset"
)


ax.set_ylabel(
    "Number of False Positives"
)


ax.set_xticklabels(
    DATASET_ORDER,
    rotation=0
)


ax.legend(
    title="Model"
)


plt.tight_layout()


FP_FIGURE = os.path.join(
    FIGURES_DIR,
    "false_positive_three_datasets.png"
)


plt.savefig(
    FP_FIGURE,
    dpi=300
)


plt.close()


# ============================================================
# 7. FN GRAPH
# ============================================================

print(
    "Generating False Negative graph..."
)


fn_plot = (
    data
    .pivot(
        index="dataset",
        columns="model",
        values="fn"
    )
    .reindex(DATASET_ORDER)
)


ax = fn_plot.plot(
    kind="bar",
    figsize=(10, 6)
)


ax.set_title(
    "False Negative Transactions"
)


ax.set_xlabel(
    "Dataset"
)


ax.set_ylabel(
    "Number of False Negatives"
)


ax.set_xticklabels(
    DATASET_ORDER,
    rotation=0
)


ax.legend(
    title="Model"
)


plt.tight_layout()


FN_FIGURE = os.path.join(
    FIGURES_DIR,
    "false_negative_three_datasets.png"
)


plt.savefig(
    FN_FIGURE,
    dpi=300
)


plt.close()


# ============================================================
# 8. COMBINED FP/FN GRAPH
# ============================================================

print(
    "Generating combined FP/FN graph..."
)


combined_plot = (
    data
    .groupby(
        ["dataset", "model"],
        observed=False
    )[
        [
            "fp",
            "fn"
        ]
    ]
    .sum()
)


ax = combined_plot.plot(
    kind="bar",
    figsize=(12, 7)
)


ax.set_title(
    "False Positive and False Negative Transactions"
)


ax.set_xlabel(
    "Dataset and Model"
)


ax.set_ylabel(
    "Number of Transactions"
)


plt.xticks(
    rotation=45,
    ha="right"
)


ax.legend(
    title="Error Type"
)


plt.tight_layout()


COMBINED_FIGURE = os.path.join(
    FIGURES_DIR,
    "false_positive_negative_three_datasets.png"
)


plt.savefig(
    COMBINED_FIGURE,
    dpi=300
)


plt.close()


# ============================================================
# 9. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 90)
print("ANALYSIS COMPLETED")
print("=" * 90)


print(
    "\nGenerated files:"
)


print(
    f"1. {OUTPUT_CSV}"
)


print(
    f"2. {DATASET_SUMMARY_FILE}"
)


print(
    f"3. {MODEL_SUMMARY_FILE}"
)


print(
    f"4. {FP_FIGURE}"
)


print(
    f"5. {FN_FIGURE}"
)


print(
    f"6. {COMBINED_FIGURE}"
)


print("\nDone.")
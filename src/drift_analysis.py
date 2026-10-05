import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DRIFT_FILE = "results/drift/adwin_drift_events.csv"

OUTPUT_DIR = "results/analysis"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "drift_analysis.csv"
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "drift_summary.txt"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 6 - ADWIN DRIFT ANALYSIS")
print("=" * 70)

print("\nLoading drift events...")

if not os.path.exists(DRIFT_FILE):
    raise FileNotFoundError(
        f"Drift file not found: {DRIFT_FILE}"
    )

df = pd.read_csv(DRIFT_FILE)

print(f"Drift events found: {len(df)}")

# Sort chronologically
df = df.sort_values(
    "transaction_index"
).reset_index(drop=True)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATA INFORMATION")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 drift events:")
print(df.head().to_string(index=False))


# ============================================================
# DISTANCE BETWEEN DRIFTS
# ============================================================

df["distance_from_previous_drift"] = (
    df["transaction_index"].diff()
)


# ============================================================
# DISTANCE TO NEXT DRIFT
# ============================================================

df["distance_to_next_drift"] = (
    df["transaction_index"].shift(-1)
    - df["transaction_index"]
)


# ============================================================
# DRIFT INTERVAL CATEGORY
# ============================================================

def classify_interval(distance):

    if pd.isna(distance):
        return "N/A"

    if distance < 1000:
        return "Very close (<1000)"

    elif distance < 5000:
        return "Close (1000-4999)"

    elif distance < 20000:
        return "Moderate (5000-19999)"

    else:
        return "Far (>=20000)"


df["drift_interval_category"] = (
    df["distance_from_previous_drift"]
    .apply(classify_interval)
)


# ============================================================
# DRIFT PERFORMANCE CHANGE
# ============================================================

# Difference between current drift performance
# and previous drift performance.

df["f1_change_from_previous"] = (
    df["f1_at_detection"].diff()
)

df["recall_change_from_previous"] = (
    df["recall_at_detection"].diff()
)

df["precision_change_from_previous"] = (
    df["precision_at_detection"].diff()
)


# ============================================================
# ADWIN WIDTH CHANGE
# ============================================================

df["adwin_width_change"] = (
    df["adwin_width"].diff()
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

total_drifts = len(df)

first_drift = df["transaction_index"].min()

last_drift = df["transaction_index"].max()

stream_span = last_drift - first_drift

mean_distance = (
    df["distance_from_previous_drift"]
    .dropna()
    .mean()
)

median_distance = (
    df["distance_from_previous_drift"]
    .dropna()
    .median()
)

minimum_distance = (
    df["distance_from_previous_drift"]
    .dropna()
    .min()
)

maximum_distance = (
    df["distance_from_previous_drift"]
    .dropna()
    .max()
)


# ============================================================
# PERFORMANCE AT DRIFT
# ============================================================

mean_accuracy = df[
    "accuracy_at_detection"
].mean()

mean_precision = df[
    "precision_at_detection"
].mean()

mean_recall = df[
    "recall_at_detection"
].mean()

mean_f1 = df[
    "f1_at_detection"
].mean()

mean_roc_auc = df[
    "roc_auc_at_detection"
].mean()


# ============================================================
# ADWIN STATISTICS
# ============================================================

mean_width = df[
    "adwin_width"
].mean()

median_width = df[
    "adwin_width"
].median()

minimum_width = df[
    "adwin_width"
].min()

maximum_width = df[
    "adwin_width"
].max()


# ============================================================
# INTERVAL COUNTS
# ============================================================

interval_counts = (
    df["drift_interval_category"]
    .value_counts()
)


# ============================================================
# SUMMARY TEXT
# ============================================================

summary = []

summary.append("=" * 70)
summary.append("ADWIN DRIFT ANALYSIS SUMMARY")
summary.append("=" * 70)

summary.append("")
summary.append(f"Total drift events       : {total_drifts}")
summary.append(f"First drift transaction  : {first_drift}")
summary.append(f"Last drift transaction   : {last_drift}")
summary.append(f"Stream span              : {stream_span}")

summary.append("")
summary.append("DRIFT INTERVAL STATISTICS")
summary.append("-" * 70)

summary.append(
    f"Mean distance            : {mean_distance:.2f}"
)

summary.append(
    f"Median distance          : {median_distance:.2f}"
)

summary.append(
    f"Minimum distance         : {minimum_distance:.2f}"
)

summary.append(
    f"Maximum distance         : {maximum_distance:.2f}"
)


summary.append("")
summary.append("PERFORMANCE AT DRIFT")
summary.append("-" * 70)

summary.append(
    f"Mean accuracy            : {mean_accuracy:.4f}"
)

summary.append(
    f"Mean precision           : {mean_precision:.4f}"
)

summary.append(
    f"Mean recall              : {mean_recall:.4f}"
)

summary.append(
    f"Mean F1                  : {mean_f1:.4f}"
)

summary.append(
    f"Mean ROC-AUC             : {mean_roc_auc:.4f}"
)


summary.append("")
summary.append("ADWIN WINDOW STATISTICS")
summary.append("-" * 70)

summary.append(
    f"Mean ADWIN width         : {mean_width:.2f}"
)

summary.append(
    f"Median ADWIN width       : {median_width:.2f}"
)

summary.append(
    f"Minimum ADWIN width      : {minimum_width:.2f}"
)

summary.append(
    f"Maximum ADWIN width      : {maximum_width:.2f}"
)


summary.append("")
summary.append("DRIFT INTERVAL CATEGORIES")
summary.append("-" * 70)

for category, count in interval_counts.items():

    percentage = (
        count / total_drifts
    ) * 100

    summary.append(
        f"{category:<25} : "
        f"{count:>3} ({percentage:.2f}%)"
    )


summary.append("")
summary.append("DRIFT EVENTS")
summary.append("-" * 70)

for _, row in df.iterrows():

    summary.append(
        f"Drift {int(row['drift_number']):>3} | "
        f"Transaction {int(row['transaction_index']):>7} | "
        f"TransactionDT {row['TransactionDT']:.1f} | "
        f"F1 {row['f1_at_detection']:.4f} | "
        f"Recall {row['recall_at_detection']:.4f} | "
        f"ADWIN width {row['adwin_width']:.1f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary)
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("DRIFT ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"\nTotal drift events: {total_drifts}"
)

print(
    f"First drift: transaction {first_drift}"
)

print(
    f"Last drift: transaction {last_drift}"
)

print(
    f"Mean distance between drifts: "
    f"{mean_distance:.2f}"
)

print(
    f"Median distance between drifts: "
    f"{median_distance:.2f}"
)

print(
    f"Mean F1 at detection: "
    f"{mean_f1:.4f}"
)

print(
    f"Mean Recall at detection: "
    f"{mean_recall:.4f}"
)

print(
    f"Mean ADWIN width: "
    f"{mean_width:.2f}"
)

print("\nSaved files:")

print(
    f"  {OUTPUT_CSV}"
)

print(
    f"  {OUTPUT_SUMMARY}"
)

print("\n" + "=" * 70)
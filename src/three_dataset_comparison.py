import os
import pandas as pd


# ============================================================
# FINAL THREE-DATASET MODEL COMPARISON
# ============================================================

print("=" * 90)
print("FINAL THREE-DATASET MODEL COMPARISON")
print("=" * 90)


# ============================================================
# SUMMARY FILES
# ============================================================

FILES = {

    # --------------------------------------------------------
    # IEEE-CIS
    # --------------------------------------------------------

    "IEEE-CIS - Baseline":
        "results/metrics/baseline_online_logistic_summary.csv",

    "IEEE-CIS - ADWIN Reset":
        "results/metrics/adaptive_adwin_logistic_summary.csv",

    "IEEE-CIS - ADWIN Replay":
        "results/metrics/improved_adaptive_logistic_summary.csv",


    # --------------------------------------------------------
    # PaySim
    # --------------------------------------------------------

    "PaySim - Baseline":
        "results/paysim/metrics/"
        "paysim_baseline_online_logistic_summary.csv",

    "PaySim - ADWIN Reset":
        "results/paysim/metrics/"
        "paysim_adaptive_adwin_reset_summary.csv",

    "PaySim - ADWIN Replay":
        "results/paysim/metrics/"
        "paysim_adaptive_adwin_replay_summary.csv",


    # --------------------------------------------------------
    # Credit Card
    # --------------------------------------------------------

    "Credit Card - Baseline":
        "results/creditcard/metrics/"
        "creditcard_baseline_online_logistic_summary.csv",

    "Credit Card - ADWIN Reset":
        "results/creditcard/metrics/"
        "creditcard_adaptive_adwin_reset_summary.csv",

    "Credit Card - ADWIN Replay":
        "results/creditcard/metrics/"
        "creditcard_adaptive_adwin_replay_summary.csv",
}


# ============================================================
# LOAD RESULTS
# ============================================================

records = []


for name, path in FILES.items():

    print(f"\nChecking: {path}")

    if not os.path.exists(path):

        print(f"WARNING: Missing file: {path}")

        continue

    try:

        df = pd.read_csv(path)

    except Exception as e:

        print(f"ERROR reading {path}: {e}")

        continue

    if len(df) == 0:

        print(f"WARNING: Empty file: {path}")

        continue

    row = df.iloc[0].to_dict()


    # ========================================================
    # DATASET
    # ========================================================

    dataset = name.split(" - ")[0]


    # ========================================================
    # MODEL
    # ========================================================

    strategy = name.split(" - ")[1]


    # ========================================================
    # THROUGHPUT
    # ========================================================

    throughput = row.get(
        "throughput",
        row.get(
            "throughput_transactions_per_second",
            0
        )
    )


    # ========================================================
    # DRIFT EVENTS
    # ========================================================

    drift_events = row.get(
        "drift_events",
        0
    )


    # Some baseline files don't contain drift_events
    if pd.isna(drift_events):

        drift_events = 0


    # ========================================================
    # CREATE RECORD
    # ========================================================

    records.append({

        "dataset":
            dataset,

        "model":
            strategy,

        "accuracy":
            float(row.get("accuracy", 0)),

        "precision":
            float(row.get("precision", 0)),

        "recall":
            float(row.get("recall", 0)),

        "f1":
            float(row.get("f1", 0)),

        "kappa":
            float(row.get("kappa", 0)),

        "fpr":
            float(row.get("fpr", 0)),

        "drift_events":
            float(drift_events),

        "transactions":
            int(float(row.get("transactions", 0))),

        "throughput":
            float(throughput)
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

results = pd.DataFrame(records)


if results.empty:

    raise RuntimeError(
        "No model results were loaded."
    )


print("\n")
print("=" * 90)
print("RAW THREE-DATASET RESULTS")
print("=" * 90)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "results/analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE COMPLETE COMPARISON
# ============================================================

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "three_dataset_model_comparison.csv"
)


results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# MODEL-LEVEL AVERAGES
# ============================================================

print("\n")
print("=" * 90)
print("AVERAGE PERFORMANCE ACROSS ALL THREE DATASETS")
print("=" * 90)


model_average = (
    results
    .groupby("model")
    [
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "kappa",
            "fpr",
            "drift_events"
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


MODEL_AVERAGE_PATH = os.path.join(
    OUTPUT_DIR,
    "three_dataset_model_averages.csv"
)


model_average.to_csv(
    MODEL_AVERAGE_PATH,
    index=False
)


# ============================================================
# BEST MODEL PER DATASET
# ============================================================

print("\n")
print("=" * 90)
print("BEST MODEL PER DATASET")
print("=" * 90)


best_records = []


for dataset in results["dataset"].unique():

    subset = results[
        results["dataset"] == dataset
    ]

    best = subset.loc[
        subset["f1"].idxmax()
    ]

    best_records.append({

        "dataset":
            dataset,

        "best_model":
            best["model"],

        "f1":
            best["f1"],

        "precision":
            best["precision"],

        "recall":
            best["recall"],

        "drift_events":
            best["drift_events"]
    })

    print(
        f"{dataset}: "
        f"{best['model']} "
        f"(F1 = {best['f1']:.4f})"
    )


BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_model_per_dataset.csv"
)


pd.DataFrame(
    best_records
).to_csv(
    BEST_MODEL_PATH,
    index=False
)


# ============================================================
# REPLAY VS RESET
# ============================================================

print("\n")
print("=" * 90)
print("REPLAY VS RESET")
print("=" * 90)


replay_comparison = []


for dataset in results["dataset"].unique():

    subset = results[
        results["dataset"] == dataset
    ]


    reset_rows = subset[
        subset["model"] == "ADWIN Reset"
    ]

    replay_rows = subset[
        subset["model"] == "ADWIN Replay"
    ]


    if reset_rows.empty or replay_rows.empty:

        continue


    reset = reset_rows.iloc[0]

    replay = replay_rows.iloc[0]


    f1_change = (
        replay["f1"]
        -
        reset["f1"]
    )


    drift_change = (
        replay["drift_events"]
        -
        reset["drift_events"]
    )


    replay_comparison.append({

        "dataset":
            dataset,

        "reset_f1":
            reset["f1"],

        "replay_f1":
            replay["f1"],

        "f1_change":
            f1_change,

        "reset_drift_events":
            reset["drift_events"],

        "replay_drift_events":
            replay["drift_events"],

        "drift_event_change":
            drift_change
    })


    print(
        f"\n{dataset}"
    )

    print(
        f"Replay F1 improvement over Reset: "
        f"{f1_change:+.4f}"
    )

    print(
        f"Replay drift-event change: "
        f"{drift_change:+.0f}"
    )


REPLAY_COMPARISON_PATH = os.path.join(
    OUTPUT_DIR,
    "replay_vs_reset_three_datasets.csv"
)


pd.DataFrame(
    replay_comparison
).to_csv(
    REPLAY_COMPARISON_PATH,
    index=False
)


# ============================================================
# DATASET-WISE F1 TABLE
# ============================================================

print("\n")
print("=" * 90)
print("F1 SCORE BY DATASET")
print("=" * 90)


f1_table = (
    results
    .pivot(
        index="dataset",
        columns="model",
        values="f1"
    )
    .reset_index()
)


print(
    f1_table.to_string(
        index=False
    )
)


F1_TABLE_PATH = os.path.join(
    OUTPUT_DIR,
    "three_dataset_f1_comparison.csv"
)


f1_table.to_csv(
    F1_TABLE_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

print(
    f"Datasets evaluated: "
    f"{results['dataset'].nunique()}"
)

print(
    f"Models evaluated: "
    f"{results['model'].nunique()}"
)

print(
    f"Total experiment rows: "
    f"{len(results)}"
)

print("\nDatasets:")

for dataset in results["dataset"].unique():

    print(
        f"  - {dataset}"
    )


# ============================================================
# FILES SAVED
# ============================================================

print("\n")
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    OUTPUT_PATH
)

print(
    MODEL_AVERAGE_PATH
)

print(
    BEST_MODEL_PATH
)

print(
    REPLAY_COMPARISON_PATH
)

print(
    F1_TABLE_PATH
)

print("\nDone.")
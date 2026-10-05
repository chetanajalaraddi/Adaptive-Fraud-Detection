import os
import pandas as pd


ANALYSIS_DIR = "results/analysis"


def test_three_dataset_comparison_exists():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    assert os.path.exists(file_path), \
        "Three-dataset comparison file was not generated"


def test_three_dataset_comparison_structure():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

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

    for column in required_columns:
        assert column in df.columns, \
            f"Missing required column: {column}"


def test_three_datasets_present():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

    expected_datasets = {
        "IEEE-CIS",
        "PaySim",
        "Credit Card"
    }

    actual_datasets = set(df["dataset"])

    assert expected_datasets.issubset(actual_datasets)


def test_three_models_present():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

    expected_models = {
        "Baseline",
        "ADWIN Reset",
        "ADWIN Replay"
    }

    actual_models = set(df["model"])

    assert expected_models.issubset(actual_models)


def test_nine_experiment_rows():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

    assert len(df) == 9


def test_metrics_are_valid():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "kappa",
        "fpr"
    ]

    for metric in metrics:

        assert df[metric].notna().all(), \
            f"{metric} contains missing values"

        assert (df[metric] >= 0).all(), \
            f"{metric} contains negative values"

        assert (df[metric] <= 1).all(), \
            f"{metric} contains values greater than 1"


def test_drift_events_are_valid():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

    assert df["drift_events"].notna().all()

    assert (df["drift_events"] >= 0).all()


def test_transaction_counts_are_valid():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

    assert (df["transactions"] > 0).all()


def test_throughput_is_valid():

    file_path = os.path.join(
        ANALYSIS_DIR,
        "three_dataset_model_comparison.csv"
    )

    df = pd.read_csv(file_path)

    assert (df["throughput"] > 0).all()
    
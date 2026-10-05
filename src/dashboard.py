import os
import time
import threading
from collections import deque

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from river import preprocessing
from river import linear_model
from river import drift


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Real-Time Fraud Transaction Monitor",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Chetana\adaptive-fraud-detection"

DATASETS = {
    "IEEE-CIS": os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "ieee_features.csv"
    ),

    "PaySim": os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "paysim_features.csv"
    ),

    "Credit Card": os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "creditcard_features.csv"
    )
}

REPORT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "live_reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# AUTOMATIC MODEL SELECTION
# ============================================================

MODEL_F1 = {
    "IEEE-CIS": {
        "Baseline": 0.169015,
        "ADWIN Reset": 0.185671,
        "ADWIN Replay": 0.168758
    },

    "PaySim": {
        "Baseline": 0.663008,
        "ADWIN Reset": 0.469999,
        "ADWIN Replay": 0.662721
    },

    "Credit Card": {
        "Baseline": 0.786813,
        "ADWIN Reset": 0.692607,
        "ADWIN Replay": 0.787679
    }
}


def select_best_model(dataset_name):

    scores = MODEL_F1[dataset_name]

    return max(
        scores,
        key=scores.get
    )


# ============================================================
# SETTINGS
# ============================================================

FRAUD_THRESHOLD = 0.20

GRAPH_VISIBLE_POINTS = 300

LIVE_TABLE_ROWS = 50

REPORT_WRITE_EVERY = 100

REPLAY_SIZE = 200

LIVE_DELAY = 0.15


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px;
        background: white;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛡️ REAL-TIME FRAUD TRANSACTION MONITOR"
)

st.caption(
    "Complete Transaction Stream • Online Learning • Concept Drift Detection"
)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    return (
        preprocessing.StandardScaler()
        | linear_model.LogisticRegression()
    )


# ============================================================
# FIND TARGET COLUMN
# ============================================================

def find_target_column(df):

    possible_targets = [
        "isFraud",
        "Class",
        "label",
        "fraud",
        "Fraud",
        "target",
        "Target"
    ]

    for column in possible_targets:

        if column in df.columns:
            return column

    return None


# ============================================================
# CONVERT LABEL
# ============================================================

def convert_label(value):

    if pd.isna(value):
        return 0

    try:

        return int(
            float(value)
        )

    except (
        ValueError,
        TypeError
    ):

        value = str(
            value
        ).strip().lower()

        if value in [
            "fraud",
            "true",
            "yes",
            "1"
        ]:

            return 1

        return 0


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    row,
    target_column
):

    features = {}

    for column, value in row.items():

        if column == target_column:
            continue

        if column in [
            "isFraud",
            "Class",
            "label",
            "fraud",
            "Fraud",
            "target",
            "Target"
        ]:
            continue

        if pd.isna(value):
            value = 0

        try:

            value = float(value)

        except (
            ValueError,
            TypeError
        ):

            value = hash(
                str(value)
            ) % 100000

        features[column] = value

    return features


# ============================================================
# CREATE LIVE GRAPH
# ============================================================

def create_graph(history):

    visible = list(history)[
        -GRAPH_VISIBLE_POINTS:
    ]

    fig = go.Figure()

    # --------------------------------------------------------
    # EMPTY GRAPH
    # --------------------------------------------------------

    if not visible:

        fig.update_layout(

            title={
                "text": "LIVE FRAUD DETECTION",
                "x": 0.5
            },

            xaxis_title="Transaction Stream",

            yaxis_title="Fraud Probability",

            yaxis=dict(
                range=[0, 1]
            ),

            height=560,

            template="plotly_white",

            margin=dict(
                l=50,
                r=30,
                t=90,
                b=50
            )
        )

        return fig

    # --------------------------------------------------------
    # FRAUD PROBABILITY
    # --------------------------------------------------------

    x = [
        item["transaction"]
        for item in visible
    ]

    probability = [
        item["probability"]
        for item in visible
    ]

    fig.add_trace(

        go.Scatter(

            x=x,

            y=probability,

            mode="lines",

            name="Fraud Probability",

            line=dict(
                width=2
            )
        )
    )

    # --------------------------------------------------------
    # GENUINE
    # --------------------------------------------------------

    genuine = [
        item
        for item in visible
        if item["result"] == "True Negative"
    ]

    if genuine:

        fig.add_trace(

            go.Scatter(

                x=[
                    item["transaction"]
                    for item in genuine
                ],

                y=[
                    item["probability"]
                    for item in genuine
                ],

                mode="markers",

                name="Genuine",

                marker=dict(
                    symbol="circle",
                    size=7
                )
            )
        )

    # --------------------------------------------------------
    # FRAUD
    # --------------------------------------------------------

    fraud = [
        item
        for item in visible
        if item["result"] == "True Positive"
    ]

    if fraud:

        fig.add_trace(

            go.Scatter(

                x=[
                    item["transaction"]
                    for item in fraud
                ],

                y=[
                    item["probability"]
                    for item in fraud
                ],

                mode="markers",

                name="Fraud",

                marker=dict(
                    symbol="diamond",
                    size=9
                )
            )
        )

    # --------------------------------------------------------
    # FALSE POSITIVE
    # --------------------------------------------------------

    false_positive = [
        item
        for item in visible
        if item["result"] == "False Positive"
    ]

    if false_positive:

        fig.add_trace(

            go.Scatter(

                x=[
                    item["transaction"]
                    for item in false_positive
                ],

                y=[
                    item["probability"]
                    for item in false_positive
                ],

                mode="markers",

                name="False Positive",

                marker=dict(
                    symbol="x",
                    size=9
                )
            )
        )

    # --------------------------------------------------------
    # FALSE NEGATIVE
    # --------------------------------------------------------

    false_negative = [
        item
        for item in visible
        if item["result"] == "False Negative"
    ]

    if false_negative:

        fig.add_trace(

            go.Scatter(

                x=[
                    item["transaction"]
                    for item in false_negative
                ],

                y=[
                    item["probability"]
                    for item in false_negative
                ],

                mode="markers",

                name="False Negative",

                marker=dict(
                    symbol="triangle-up",
                    size=9
                )
            )
        )

    # --------------------------------------------------------
    # TRANSACTIONS FLAGGED
    # --------------------------------------------------------

    blocked = [
        item
        for item in visible
        if item["result"] == "Blocked"
    ]

    if blocked:

        fig.add_trace(

            go.Scatter(

                x=[
                    item["transaction"]
                    for item in blocked
                ],

                y=[
                    item["probability"]
                    for item in blocked
                ],

                mode="markers",

                name="Transactions Flagged",

                marker=dict(
                    symbol="square",
                    size=8
                )
            )
        )

    # --------------------------------------------------------
    # FRAUD THRESHOLD
    # --------------------------------------------------------

    fig.add_hline(

        y=FRAUD_THRESHOLD,

        line_dash="dash",

        annotation_text=(
            "Fraud Alert when Probability ≥ 20%"
        ),

        annotation_position="top left"
    )

    # --------------------------------------------------------
    # ADWIN DRIFT
    # --------------------------------------------------------

    drift_items = [
        item
        for item in visible
        if item["drift"]
    ]

    for item in drift_items:

        fig.add_vline(

            x=item["transaction"],

            line_dash="dot",

            line_width=2,

            annotation_text=(
                "ADWIN Drift Detected → "
                "Model Adaptation"
            ),

            annotation_position="top"
        )

    # --------------------------------------------------------
    # GRAPH LAYOUT
    # --------------------------------------------------------

    fig.update_layout(

        title={
            "text": "LIVE FRAUD DETECTION",
            "x": 0.5
        },

        xaxis_title="Transaction Stream",

        yaxis_title="Fraud Probability",

        yaxis=dict(
            range=[0, 1]
        ),

        height=560,

        template="plotly_white",

        hovermode="x unified",

        legend=dict(

            orientation="h",

            yanchor="bottom",

            y=1.02,

            xanchor="left",

            x=0
        ),

        margin=dict(
            l=50,
            r=30,
            t=100,
            b=50
        )
    )

    return fig


# ============================================================
# KPI DISPLAY
# ============================================================

def render_kpis(runtime):

    if runtime["running"]:

        status = "● RUNNING"

    elif runtime["paused"]:

        status = "⏸ PAUSED"

    elif runtime["completed"]:

        status = "● COMPLETED"

    elif runtime["stopped"]:

        status = "⏹ STOPPED"

    else:

        status = "● READY"

    cols = st.columns(6)

    cols[0].metric(
        "STATUS",
        status
    )

    cols[1].metric(
        "TRANSACTIONS",
        f'{runtime["transactions"]:,}'
    )

    cols[2].metric(
        "FRAUD DETECTED",
        f'{runtime["fraud_detected"]:,}'
    )

    cols[3].metric(
        "ACCOUNTS FLAGGED",
        f'{len(runtime["flagged_accounts"]):,}'
    )

    cols[4].metric(
        "TRANSACTIONS FLAGGED",
        f'{runtime["transactions_flagged"]:,}'
    )

    cols[5].metric(
        "DRIFT EVENTS",
        f'{runtime["drift_events"]:,}'
    )


# ============================================================
# LIVE TRANSACTION RESULTS
# ============================================================

def render_live_results(runtime):

    st.subheader(
        "📄 LIVE TRANSACTION RESULTS"
    )

    rows = list(
        runtime["live_rows"]
    )[-LIVE_TABLE_ROWS:]

    if not rows:

        st.info(
            "Transactions will appear here when detection starts."
        )

        return

    df = pd.DataFrame(
        rows
    )

    display_columns = [

        "Transaction",

        "Dataset",

        "Model Prediction",

        "Ground Truth",

        "Result",

        "Fraud Probability",

        "Drift",

        "Account Status"
    ]

    df = df[
        [
            column
            for column in display_columns
            if column in df.columns
        ]
    ]

    st.dataframe(

        df,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# WRITE REPORT
# ============================================================

def write_report(runtime):

    if not runtime["report_rows"]:
        return

    report_path = os.path.join(

        REPORT_DIR,

        "complete_transaction_report.csv"
    )

    pd.DataFrame(

        runtime["report_rows"]

    ).to_csv(

        report_path,

        index=False
    )

    runtime["report_path"] = report_path


# ============================================================
# CREATE RUNTIME
# ============================================================

def create_runtime():

    return {

        "running": False,

        "paused": False,

        "completed": False,

        "stopped": False,

        "transactions": 0,

        "fraud_detected": 0,

        "transactions_flagged": 0,

        "drift_events": 0,

        "flagged_accounts": set(),

        "history": deque(
            maxlen=GRAPH_VISIBLE_POINTS
        ),

        "live_rows": deque(
            maxlen=LIVE_TABLE_ROWS
        ),

        "report_rows": [],

        "report_path": None,

        # Backend values only.
        # They are NOT displayed.

        "current_dataset": None,

        "current_model": None,

        "dataset_transaction": 0,

        "pause_event": threading.Event(),

        "stop_event": threading.Event(),

        "thread": None,

        "error": None
    }


# ============================================================
# PROCESS ONE TRANSACTION
# ============================================================

def process_transaction(

    runtime,

    model,

    detector,

    replay_buffer,

    row,

    target_column,

    dataset_name,

    model_name,

    transaction_number,

    account_value

):

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    features = prepare_features(

        row,

        target_column
    )

    actual_label = convert_label(

        row[target_column]
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba_one(

        features
    )

    probability = float(

        probabilities.get(
            1,
            0.0
        )
    )

    prediction = int(

        probability >= FRAUD_THRESHOLD
    )

    # --------------------------------------------------------
    # INTERNAL ACCOUNT KEY
    # --------------------------------------------------------

    account_key = (

        dataset_name,

        str(account_value)
    )

    already_flagged = (

        account_key
        in runtime["flagged_accounts"]
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if already_flagged:

        result = "Blocked"

        runtime["transactions_flagged"] += 1

        account_status = "Flagged"

    else:

        # TRUE POSITIVE
        if (

            prediction == 1

            and

            actual_label == 1

        ):

            result = "True Positive"

            runtime["fraud_detected"] += 1

            runtime["flagged_accounts"].add(

                account_key
            )

            account_status = "Flagged"

        # TRUE NEGATIVE
        elif (

            prediction == 0

            and

            actual_label == 0

        ):

            result = "True Negative"

            account_status = "Active"

        # FALSE POSITIVE
        elif (

            prediction == 1

            and

            actual_label == 0

        ):

            result = "False Positive"

            account_status = "Active"

        # FALSE NEGATIVE
        else:

            result = "False Negative"

            account_status = "Active"

    # --------------------------------------------------------
    # ADWIN
    # --------------------------------------------------------

    correctness = int(

        prediction == actual_label
    )

    detector.update(

        correctness
    )

    drift_detected = bool(

        detector.drift_detected
    )

    if drift_detected:

        runtime["drift_events"] += 1

    # --------------------------------------------------------
    # ONLINE LEARNING
    # --------------------------------------------------------

    model.learn_one(

        features,

        actual_label
    )

    replay_buffer.append(

        (
            features,
            actual_label
        )
    )

    # --------------------------------------------------------
    # MODEL ADAPTATION
    # --------------------------------------------------------

    if (

        drift_detected

        and

        model_name == "ADWIN Reset"

    ):

        model = create_model()

    elif (

        drift_detected

        and

        model_name == "ADWIN Replay"

    ):

        model = create_model()

        for (

            replay_features,
            replay_label

        ) in replay_buffer:

            model.learn_one(

                replay_features,

                replay_label
            )

    # --------------------------------------------------------
    # GRAPH HISTORY
    # --------------------------------------------------------

    runtime["history"].append(

        {

            "transaction":
                transaction_number,

            "probability":
                probability,

            "result":
                result,

            "drift":
                drift_detected
        }
    )

    # --------------------------------------------------------
    # LIVE TABLE
    # --------------------------------------------------------

    runtime["live_rows"].append(

        {

            "Transaction":
                transaction_number,

            "Dataset":
                dataset_name,

            "Model Prediction":
                (
                    "Fraud"
                    if prediction == 1
                    else "Genuine"
                ),

            "Ground Truth":
                (
                    "Fraud"
                    if actual_label == 1
                    else "Genuine"
                ),

            "Result":
                result,

            "Fraud Probability":
                round(
                    probability,
                    4
                ),

            "Drift":
                (
                    "Detected"
                    if drift_detected
                    else "-"
                ),

            "Account Status":
                account_status
        }
    )

    # --------------------------------------------------------
    # COMPLETE REPORT
    # --------------------------------------------------------

    runtime["report_rows"].append(

        {

            "Transaction":
                transaction_number,

            "Dataset":
                dataset_name,

            "Model":
                model_name,

            "Model Prediction":
                (
                    "Fraud"
                    if prediction == 1
                    else "Genuine"
                ),

            "Ground Truth":
                (
                    "Fraud"
                    if actual_label == 1
                    else "Genuine"
                ),

            "Result":
                result,

            "Fraud Probability":
                round(
                    probability,
                    6
                ),

            "Drift":
                (
                    "Detected"
                    if drift_detected
                    else "No"
                ),

            "Account Status":
                account_status
        }
    )

    return model


# ============================================================
# PROCESS ALL TRANSACTIONS
# ============================================================

def process_transactions(runtime):

    try:

        runtime["running"] = True

        runtime["paused"] = False

        runtime["completed"] = False

        runtime["stopped"] = False

        # ====================================================
        # DATASET LOOP
        # ====================================================

        for (

            dataset_name,

            dataset_path

        ) in DATASETS.items():

            # ------------------------------------------------
            # STOP
            # ------------------------------------------------

            if runtime["stop_event"].is_set():

                break

            runtime["current_dataset"] = dataset_name

            # ------------------------------------------------
            # AUTOMATIC MODEL SELECTION
            # ------------------------------------------------

            model_name = select_best_model(

                dataset_name
            )

            runtime["current_model"] = model_name

            # ------------------------------------------------
            # CREATE MODEL
            # ------------------------------------------------

            model = create_model()

            # ------------------------------------------------
            # CREATE ADWIN
            # ------------------------------------------------

            detector = drift.ADWIN()

            # ------------------------------------------------
            # REPLAY BUFFER
            # ------------------------------------------------

            replay_buffer = deque(

                maxlen=REPLAY_SIZE
            )

            # ------------------------------------------------
            # DATASET CHECK
            # ------------------------------------------------

            if not os.path.exists(

                dataset_path
            ):

                runtime["report_rows"].append(

                    {

                        "Dataset":
                            dataset_name,

                        "Error":
                            "Dataset not found: "
                            + dataset_path
                    }
                )

                continue

            # ------------------------------------------------
            # FIND TARGET
            # ------------------------------------------------

            first_chunk = pd.read_csv(

                dataset_path,

                nrows=5,

                low_memory=False
            )

            target_column = find_target_column(

                first_chunk
            )

            if target_column is None:

                runtime["report_rows"].append(

                    {

                        "Dataset":
                            dataset_name,

                        "Error":
                            "Fraud target column not found"
                    }
                )

                continue

            # ------------------------------------------------
            # ACCOUNT COLUMN
            # ------------------------------------------------

            account_candidates = [

                "account_id",
                "AccountID",
                "account",
                "Account",
                "nameOrig",
                "card1",
                "card"
            ]

            account_column = None

            for candidate in account_candidates:

                if candidate in first_chunk.columns:

                    account_column = candidate

                    break

            # =================================================
            # READ COMPLETE DATASET
            # =================================================

            for chunk in pd.read_csv(

                dataset_path,

                chunksize=5000,

                low_memory=False

            ):

                # =============================================
                # TRANSACTION-BY-TRANSACTION PROCESSING
                # =============================================

                for _, row in chunk.iterrows():

                    # =========================================
                    # STOP
                    # =========================================

                    if runtime["stop_event"].is_set():

                        runtime["running"] = False

                        runtime["paused"] = False

                        runtime["stopped"] = True

                        write_report(
                            runtime
                        )

                        return

                    # =========================================
                    # PAUSE
                    # =========================================

                    while runtime["pause_event"].is_set():

                        runtime["running"] = False

                        runtime["paused"] = True

                        if runtime["stop_event"].is_set():

                            runtime["running"] = False

                            runtime["paused"] = False

                            runtime["stopped"] = True

                            write_report(
                                runtime
                            )

                            return

                        time.sleep(
                            0.1
                        )

                    # =========================================
                    # RESUME
                    # =========================================

                    runtime["running"] = True

                    runtime["paused"] = False

                    # =========================================
                    # ACCOUNT VALUE
                    # =========================================

                    if account_column is not None:

                        account_value = row.get(

                            account_column,

                            "unknown"
                        )

                    else:

                        account_value = (

                            dataset_name
                            + "_"
                            + str(
                                runtime["transactions"]
                            )
                        )

                    # =========================================
                    # TRANSACTION NUMBER
                    # =========================================

                    runtime["transactions"] += 1

                    runtime["dataset_transaction"] += 1

                    transaction_number = (

                        runtime["transactions"]
                    )

                    # =========================================
                    # PROCESS
                    # =========================================

                    model = process_transaction(

                        runtime,

                        model,

                        detector,

                        replay_buffer,

                        row,

                        target_column,

                        dataset_name,

                        model_name,

                        transaction_number,

                        account_value
                    )

                    # =========================================
                    # WRITE REPORT
                    # =========================================

                    if (

                        runtime["transactions"]

                        % REPORT_WRITE_EVERY

                        == 0

                    ):

                        write_report(
                            runtime
                        )

                    # =========================================
                    # LIVE DELAY
                    # =========================================

                    time.sleep(
                        LIVE_DELAY
                    )

        # ====================================================
        # COMPLETED
        # ====================================================

        runtime["running"] = False

        runtime["paused"] = False

        if runtime["stop_event"].is_set():

            runtime["stopped"] = True

            runtime["completed"] = False

        else:

            runtime["stopped"] = False

            runtime["completed"] = True

        write_report(
            runtime
        )

    except Exception as error:

        runtime["running"] = False

        runtime["paused"] = False

        runtime["stopped"] = True

        runtime["error"] = str(
            error
        )

        write_report(
            runtime
        )


# ============================================================
# SESSION STATE
# ============================================================

if "runtime" not in st.session_state:

    st.session_state.runtime = create_runtime()

runtime = st.session_state.runtime


# ============================================================
# DETECTION CONTROL
# ============================================================

st.subheader(
    "🎛️ DETECTION CONTROL"
)


# ============================================================
# START
# ============================================================

if (

    not runtime["running"]

    and

    not runtime["paused"]

):

    if st.button(

        "▶ START DETECTION",

        type="primary",

        use_container_width=True

    ):

        runtime = create_runtime()

        st.session_state.runtime = runtime

        worker = threading.Thread(

            target=process_transactions,

            args=(runtime,),

            daemon=True
        )

        runtime["thread"] = worker

        worker.start()

        st.rerun()


# ============================================================
# RUNNING
# ============================================================

elif runtime["running"]:

    cols = st.columns(3)

    with cols[0]:

        if st.button(

            "⏸ PAUSE",

            use_container_width=True

        ):

            runtime["pause_event"].set()

            st.rerun()

    with cols[1]:

        if st.button(

            "⏹ STOP",

            use_container_width=True

        ):

            runtime["stop_event"].set()

            runtime["pause_event"].clear()

            st.rerun()

    with cols[2]:

        st.button(

            "● RUNNING",

            disabled=True,

            use_container_width=True

        )


# ============================================================
# PAUSED
# ============================================================

elif runtime["paused"]:

    cols = st.columns(3)

    with cols[0]:

        if st.button(

            "▶ RESUME",

            type="primary",

            use_container_width=True

        ):

            runtime["pause_event"].clear()

            st.rerun()

    with cols[1]:

        if st.button(

            "⏹ STOP",

            use_container_width=True

        ):

            runtime["stop_event"].set()

            runtime["pause_event"].clear()

            st.rerun()

    with cols[2]:

        st.button(

            "⏸ PAUSED",

            disabled=True,

            use_container_width=True

        )


# ============================================================
# KPI
# ============================================================

render_kpis(
    runtime
)


# ============================================================
# ONLY ONE GRAPH
# ============================================================

st.subheader(
    "📈 LIVE FRAUD DETECTION"
)

st.plotly_chart(
    create_graph(
        runtime["history"]
    ),
    use_container_width=True
)


# ============================================================
# LIVE TRANSACTION RESULTS
# ============================================================

render_live_results(
    runtime
)


# ============================================================
# DOWNLOAD COMPLETE REPORT
# ============================================================

report_path = runtime.get(
    "report_path"
)

if (

    report_path

    and

    os.path.exists(
        report_path
    )

):

    with open(
        report_path,
        "rb"
    ) as file:

        st.download_button(

            label=(
                "⬇️ DOWNLOAD COMPLETE "
                "TRANSACTION REPORT"
            ),

            data=file.read(),

            file_name=(
                "complete_transaction_report.csv"
            ),

            mime="text/csv",

            use_container_width=True
        )


# ============================================================
# ERROR
# ============================================================

if runtime.get(
    "error"
):

    st.error(

        "Detection stopped because of an error: "

        + str(
            runtime["error"]
        )
    )


# ============================================================
# AUTO REFRESH
# ============================================================

if (

    runtime["running"]

    or

    runtime["paused"]

):

    time.sleep(
        0.5
    )

    st.rerun()
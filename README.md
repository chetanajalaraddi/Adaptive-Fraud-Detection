# Adaptive Fraud Detection

### Real-Time Financial Fraud Detection Using Incremental Learning

## About the Project

This project is about detecting fraudulent financial transactions in real time.

Normally, a fraud detection model is trained using old transaction data. But transaction patterns can change over time. Because of this, a model that works well today may not work as well later.

To solve this problem, we are developing an **adaptive fraud detection system**. The system processes transactions one by one and updates the model as new data comes in.

The project also detects **concept drift**, which means that the pattern of the incoming data has changed.

## What We Are Trying to Do

The main goals of this project are:

* Detect fraudulent transactions.
* Process transactions one by one.
* Detect changes in transaction patterns.
* Update the model when the data pattern changes.
* Reduce false fraud alerts.
* Compare different machine learning models.
* Test the system on different fraud datasets.
* Show the results using a simple dashboard.

## How Our System Works

The basic working of our system is:

```text
Transaction
     ↓
Data Preprocessing
     ↓
Feature Extraction
     ↓
Online Machine Learning Model
     ↓
Fraud / Normal Prediction
     ↓
Check for Concept Drift
     ↓
Update Model
     ↓
Show Results on Dashboard
```

The system continues this process as new transactions arrive.

## Machine Learning Models

We have worked with different models and approaches for comparison.

Some of them are:

* Adaptive Random Forest
* Hoeffding Adaptive Tree
* Online Logistic Regression
* Naive Bayes
* Baseline Model
* ADWIN Reset
* ADWIN Replay

The main purpose of comparing these methods is to understand which approach works better when transaction data changes.

## Concept Drift

**Concept drift** means that the pattern in the data changes over time.

For example, fraudsters may start using a different transaction pattern. If the model only learns from old data, it may not detect the new type of fraud properly.

To detect these changes, we use:

* ADWIN
* DDM

When a change is detected, the adaptive system can update the model based on the new data.

## Datasets

We are testing the project using different financial fraud datasets:

### IEEE-CIS Fraud Detection

A large dataset containing online transaction information. This is one of the main datasets used in our project.

### PaySim

A simulated financial transaction dataset used for testing fraud detection methods.

### Credit Card Fraud Detection

A dataset containing credit card transactions, including fraudulent and normal transactions.

The original datasets are **not uploaded to this public GitHub repository** because some of them are very large.

## Performance Measures

We use different measures to check how well the models perform:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Kappa
* False Positive Rate
* Number of drift events
* Processing speed

For fraud detection, we mainly look at **precision, recall, F1-score and false positive rate**, because fraud transactions are much fewer than normal transactions.

## Dashboard

We have also developed a **Streamlit dashboard** to show the fraud detection process.

The dashboard is used to display things such as:

* Number of transactions processed
* Fraud detected
* Model performance
* Drift events
* Fraud alerts
* Live performance
* Graphs and analysis

## Project Structure

```text
Adaptive-Fraud-Detection/
│
├── .gitignore
├── README.md
├── requirements.txt
├── project_structure.txt
├── check_dataset.py
├── check_identity.py
│
└── src/
    ├── adaptive_model.py
    ├── compare_models.py
    ├── creditcard_adaptive_replay.py
    ├── creditcard_adaptive_reset.py
    ├── creditcard_baseline.py
    ├── creditcard_preprocess.py
    ├── dashboard.py
    ├── drift_analysis.py
    ├── drift_detection.py
    ├── drift_performance_analysis.py
    ├── false_positive_negative_analysis.py
    ├── feature_engineering.py
    ├── final_three_dataset_analysis.py
    ├── improved_adaptive_engine.py
    ├── paysim_adaptive_replay.py
    ├── paysim_adaptive_reset.py
    ├── paysim_baseline_online.py
    ├── paysim_preprocess.py
    ├── preprocess.py
    ├── prequential_model.py
    ├── streaming_engine.py
    ├── three_dataset_comparison.py
    ├── threshold_analysis.py
    ├── threshold_validation.py
    ├── window_analysis.py
    │
    └── tests/
        ├── test_adwin.py
        ├── test_data_loading.py
        ├── test_metrics.py
        ├── test_model.py
        └── test_results.py
```

## Technologies Used

* **Python** – Main programming language
* **River** – Online machine learning
* **Streamlit** – Dashboard
* **Pandas** – Data processing
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning and evaluation
* **Plotly** – Graphs and visualization
* **Git & GitHub** – Project version control and teamwork

## How to Run the Project

First, clone the repository:

```bash
git clone https://github.com/chetanajalaraddi/Adaptive-Fraud-Detection.git
```

Go to the project folder:

```bash
cd Adaptive-Fraud-Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

To start the dashboard:

```bash
streamlit run src/dashboard.py
```

## Dataset

The large datasets are not included in this repository.

The following types of files are ignored:

```text
data/
*.csv
*.xlsx
venv/
.env
results/
```

The datasets need to be downloaded separately and placed in the required folder.

## Team Work

We are using GitHub to work together on this project.

Each team member can work on their own changes and push them to GitHub.

Our basic workflow is:

```text
Create / Change Code
        ↓
Test the Changes
        ↓
git add .
        ↓
git commit
        ↓
git push
        ↓
GitHub
```

## Project Status

**Project:** Adaptive Fraud Detection

**Area:** Machine Learning and Fraud Detection

**Main Focus:** Online Learning and Concept Drift

## Future Work

In the future, we plan to improve the project by:

* Improving the adaptive model
* Reducing false fraud alerts
* Improving drift detection
* Adding more datasets
* Improving the dashboard
* Adding better real-time transaction processing
* Studying more ways to explain why a transaction is marked as fraud

## GitHub Repository

https://github.com/chetanajalaraddi/Adaptive-Fraud-Detection

---

This project is being developed as an academic project to study real-time fraud detection and machine learning models that can adapt to changing transaction patterns.

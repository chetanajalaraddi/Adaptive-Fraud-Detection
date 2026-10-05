import pandas as pd

# Load training transaction data
train = pd.read_csv("data/raw/train_transaction.csv")

# 1. Shape
print("\nSHAPE:")
print(train.shape)

# 2. Fraud distribution
print("\nFRAUD DISTRIBUTION:")
print(train["isFraud"].value_counts())

# 3. Fraud percentage
print("\nFRAUD DISTRIBUTION (%):")
print(train["isFraud"].value_counts(normalize=True) * 100)

# 4. Columns
print("\nCOLUMNS:")
for col in train.columns:
    print(col)
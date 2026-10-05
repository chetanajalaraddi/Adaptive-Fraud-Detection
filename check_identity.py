import pandas as pd

identity = pd.read_csv("data/raw/train_identity.csv")

print("\nIDENTITY SHAPE:")
print(identity.shape)

print("\nIDENTITY COLUMNS:")
for col in identity.columns:
    print(col)

print("\nFIRST 5 ROWS:")
print(identity.head())
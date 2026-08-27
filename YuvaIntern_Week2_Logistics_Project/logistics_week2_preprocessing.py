"""Yuva Intern - Week 2
Data Collection, Cleaning and Preprocessing for Logistics Analysis

This script is a reusable preprocessing template for the public
DataCo Smart Supply Chain dataset. Column checks are conditional so the
script remains safe if the CSV version has slightly different names.
"""
import pandas as pd
import numpy as np

DATA_PATH = "DataCoSupplyChainDataset.csv"

# 1. Load data
# latin1 is commonly needed for this dataset's text fields.
df = pd.read_csv(DATA_PATH, encoding="latin1")
print("Original shape:", df.shape)

# 2. Initial inspection
print(df.head())
print(df.dtypes)
print("Duplicate rows:", df.duplicated().sum())
missing_pct = df.isna().mean().mul(100).sort_values(ascending=False)
print("Top missing-value percentages:")
print(missing_pct.head(15))

# 3. Remove exact duplicates after validation
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# 4. Safe type conversion
if "shipping date (DateOrders)" in df.columns:
    df["shipping date (DateOrders)"] = pd.to_datetime(
        df["shipping date (DateOrders)"], errors="coerce"
    )

if "Order Item Quantity" in df.columns:
    df["Order Item Quantity"] = pd.to_numeric(
        df["Order Item Quantity"], errors="coerce"
    )

# 5. Missing-value handling examples
if "Order Item Quantity" in df.columns:
    df["Order Item Quantity"] = df["Order Item Quantity"].fillna(
        df["Order Item Quantity"].median()
    )

if "Shipping Mode" in df.columns:
    df["Shipping Mode"] = df["Shipping Mode"].fillna("Unknown")
    df["Shipping Mode"] = (
        df["Shipping Mode"].astype("string").str.strip().str.lower()
    )

# 6. IQR outlier flag (review, do not automatically delete)
col = "Days for shipping (real)"
if col in df.columns:
    series = pd.to_numeric(df[col], errors="coerce")
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (series < lower) | (series > upper)
    print(f"Potential {col} outliers:", int(outlier_mask.sum()))

# 7. Final validation
print("Final shape:", df.shape)
print("Duplicates remaining:", int(df.duplicated().sum()))
print("Missing cells remaining:", int(df.isna().sum().sum()))
print("Duplicates removed:", duplicates_removed)

# The cleaned dataframe can now be exported for Week 3 analysis.
df.to_csv("DataCoSupplyChain_cleaned.csv", index=False)
print("Saved: DataCoSupplyChain_cleaned.csv")

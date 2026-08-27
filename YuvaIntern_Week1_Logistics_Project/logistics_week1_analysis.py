"""Yuva Intern - Logistics Data Analyst - Week 1
Strategic Planning and Data Exploration in Logistics

This script is a baseline implementation illustrating data loading, cleaning,
KPI creation, exploratory analysis, regression, and clustering.

Dataset expected: DataCoSupplyChainDataset.csv
Source: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

DATA_PATH = "DataCoSupplyChainDataset.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    print("Shape:", df.shape)
    print(df.head())
    print("Duplicate rows:", df.duplicated().sum())
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()

    date_cols = ["shipping date (DateOrders)", "order date (DateOrders)"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    required = {"Days for shipping (real)", "Days for shipment (scheduled)"}
    if required.issubset(df.columns):
        df["delay_days"] = (
            df["Days for shipping (real)"]
            - df["Days for shipment (scheduled)"]
        )
        df["on_time"] = (df["delay_days"] <= 0).astype(int)

    return df


def calculate_kpis(df: pd.DataFrame) -> None:
    if "on_time" in df.columns:
        print(f"On-time delivery rate: {df['on_time'].mean() * 100:.2f}%")
    if "delay_days" in df.columns:
        print(f"Average delivery delay: {df['delay_days'].mean():.2f} days")
    if {"Order Item Total"}.issubset(df.columns):
        print(f"Average order value: {df['Order Item Total'].mean():.2f}")


def run_eda(df: pd.DataFrame) -> None:
    if "delay_days" not in df.columns:
        return

    df["delay_days"].dropna().plot(kind="hist", bins=20)
    plt.xlabel("Delivery delay (days)")
    plt.ylabel("Number of orders")
    plt.title("Distribution of Delivery Delay")
    plt.tight_layout()
    plt.show()

    if "Shipping Mode" in df.columns:
        mode_delay = df.groupby("Shipping Mode")["delay_days"].mean().sort_values()
        mode_delay.plot(kind="bar")
        plt.ylabel("Average delay (days)")
        plt.title("Average Delay by Shipping Mode")
        plt.tight_layout()
        plt.show()


def run_regression(df: pd.DataFrame) -> None:
    required = {
        "delay_days",
        "Days for shipment (scheduled)",
        "Shipping Mode",
        "Market",
        "Order Region",
    }
    if not required.issubset(df.columns):
        print("Regression skipped: required columns are missing.")
        return

    features = [
        "Days for shipment (scheduled)",
        "Shipping Mode",
        "Market",
        "Order Region",
    ]
    model_df = df.dropna(subset=["delay_days"])[features + ["delay_days"]].copy()

    X = model_df[features]
    y = model_df["delay_days"]

    num_features = ["Days for shipment (scheduled)"]
    cat_features = ["Shipping Mode", "Market", "Order Region"]

    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    model = Pipeline(
        [("preprocess", preprocess), ("regression", LinearRegression())]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("Regression MAE:", mean_absolute_error(y_test, pred))
    print("Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("Regression R2:", r2_score(y_test, pred))


def run_clustering(df: pd.DataFrame) -> None:
    required = {"Order Item Quantity", "Order Item Total", "delay_days"}
    if not required.issubset(df.columns):
        print("Clustering skipped: required columns are missing.")
        return

    cluster_df = df[[
        "Order Item Quantity",
        "Order Item Total",
        "delay_days",
    ]].dropna().copy()

    X_scaled = StandardScaler().fit_transform(cluster_df)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    cluster_df["cluster"] = kmeans.fit_predict(X_scaled)

    print("Cluster summary:")
    print(cluster_df.groupby("cluster").mean(numeric_only=True))


def main() -> None:
    df = load_data(DATA_PATH)
    df = prepare_data(df)
    calculate_kpis(df)
    run_eda(df)
    run_regression(df)
    run_clustering(df)


if __name__ == "__main__":
    main()

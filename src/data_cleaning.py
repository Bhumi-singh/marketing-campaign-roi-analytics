"""
data_cleaning.py
Cleans the raw Bank Marketing dataset and saves processed output.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "../data/raw/bank-additional/bank-additional/bank-additional-full.csv"
OUT_PATH = "../data/processed/cleaned_data.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "y": "subscribed",
        "emp.var.rate": "emp_var_rate",
        "cons.price.idx": "cons_price_idx",
        "cons.conf.idx": "cons_conf_idx",
        "nr.employed": "nr_employed",
        "euribor3m": "euribor3m",
    })
    df.columns = [c.replace(".", "_") for c in df.columns]
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df["subscribed"] = df["subscribed"].map({"yes": 1, "no": 0})
    return df


def handle_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    unknown_cols = ["job", "marital", "education", "default", "housing", "loan"]
    for col in unknown_cols:
        if col in df.columns:
            mode_val = df[df[col] != "unknown"][col].mode()[0]
            df[col] = df[col].replace("unknown", mode_val)
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = df["age"].astype(int)
    df["campaign"] = df["campaign"].astype(int)
    df["pdays"] = df["pdays"].astype(int)
    df["previous"] = df["previous"].astype(int)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Cap campaign contacts at 99th percentile
    cap = df["campaign"].quantile(0.99)
    df["campaign"] = df["campaign"].clip(upper=cap)
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Age bands
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"]
    )
    # Contact recency flag
    df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
    # High frequency contact flag (>3 times)
    df["high_contact_freq"] = (df["campaign"] > 3).astype(int)
    # Season from month
    season_map = {
        "mar": "Spring", "apr": "Spring", "may": "Spring",
        "jun": "Summer", "jul": "Summer", "aug": "Summer",
        "sep": "Autumn", "oct": "Autumn", "nov": "Autumn",
        "dec": "Winter", "jan": "Winter", "feb": "Winter"
    }
    df["season"] = df["month"].map(season_map)
    return df


def clean(path: str = RAW_PATH, out: str = OUT_PATH) -> pd.DataFrame:
    df = load_raw(path)
    df = rename_columns(df)
    df = encode_target(df)
    df = handle_unknowns(df)
    df = fix_dtypes(df)
    df = remove_outliers(df)
    df = feature_engineer(df)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Cleaned data saved → {out}  ({df.shape[0]:,} rows)")
    return df


if __name__ == "__main__":
    clean()

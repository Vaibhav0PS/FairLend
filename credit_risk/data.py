"""Loading and cleaning the raw credit risk dataset."""

import hashlib
from pathlib import Path

import pandas as pd

from .config import FEATURE_COLS, MAX_AGE, MAX_EMP_LENGTH, MIN_AGE, TARGET


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURE_COLS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and impossible rows.

    Missing employment length and interest rate are kept; the model's imputer
    fills them. (The notebook's `emp_length <= age` filter silently dropped
    every row with a missing employment length.)
    """
    df = df.drop_duplicates()
    df = df[df["person_age"].between(MIN_AGE, MAX_AGE)]

    emp = df["person_emp_length"]
    df = df[emp.isna() | ((emp <= MAX_EMP_LENGTH) & (emp <= df["person_age"]))]

    df = df[(df["person_income"] > 0) & (df["loan_amnt"] > 0)]
    return df.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS].copy(), df[TARGET].astype(int)

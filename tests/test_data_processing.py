import pandas as pd
import pytest

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


@pytest.fixture
def raw_df():
    """A tiny synthetic frame that mimics the raw UCI file, including a
    couple of missing values encoded as '?'."""
    data = [
        [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, "0", "1", 0],
        [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, "0", "2", 1],
        [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, "?", "2", 0],
        [56, 1, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, "0", "?", 2],
    ]
    return pd.DataFrame(data, columns=COLUMN_NAMES)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the cleaning logic used in the notebook."""
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df["target"] = (df["target"] > 0).astype(int)
    return df


def test_missing_values_are_coerced_to_nan(raw_df):
    df = raw_df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    assert df["ca"].isna().sum() == 1
    assert df["thal"].isna().sum() == 1


def test_clean_removes_all_missing_values(raw_df):
    cleaned = clean(raw_df)
    assert cleaned.isna().sum().sum() == 0


def test_target_is_binarized(raw_df):
    cleaned = clean(raw_df)
    assert set(cleaned["target"].unique()) <= {0, 1}
    # row with original target=2 should become 1
    assert cleaned.loc[3, "target"] == 1


def test_no_duplicate_rows_after_cleaning(raw_df):
    cleaned = clean(raw_df)
    assert cleaned.duplicated().sum() == 0


def test_expected_columns_present(raw_df):
    cleaned = clean(raw_df)
    assert list(cleaned.columns) == COLUMN_NAMES


def test_feature_ranges_are_plausible(raw_df):
    cleaned = clean(raw_df)
    assert cleaned["age"].between(0, 120).all()
    assert cleaned["thalach"].between(0, 250).all()

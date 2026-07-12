"""
Train the heart-disease risk model and save the pipeline + metadata.

This mirrors the notebook's Data Acquisition -> Cleaning -> Feature
Engineering -> Model Development -> Packaging steps in a script form,
so it can run headlessly in CI/CD (GitHub Actions) or from the command line.

Usage:
    python scripts/train.py                # full grid search
    python scripts/train.py --fast         # tiny grid, for quick CI runs
"""
import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
MIRROR_URL = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal", "ca"]
BINARY_COLS = ["sex", "fbs", "exang"]


def download_data(data_dir: str) -> pd.DataFrame:
    raw_path = os.path.join(data_dir, "heart_disease_raw.csv")
    try:
        df = pd.read_csv(UCI_URL, header=None, names=COLUMN_NAMES, na_values="?")
        if df.shape[0] > 0:
            df.to_csv(raw_path, index=False)
            return df
    except Exception as e:
        print("UCI archive fetch failed:", e)

    try:
        from ucimlrepo import fetch_ucirepo
        hd = fetch_ucirepo(id=45)
        df = pd.concat([hd.data.features, hd.data.targets], axis=1)
        df.columns = COLUMN_NAMES
        df.to_csv(raw_path, index=False)
        return df
    except Exception as e:
        print("ucimlrepo fetch failed:", e)

    df = pd.read_csv(MIRROR_URL)
    df.columns = COLUMN_NAMES
    df.to_csv(raw_path, index=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df["target"] = (df["target"] > 0).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), CONTINUOUS_COLS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("binary", "passthrough", BINARY_COLS),
        ]
    )


def get_model_grids(preprocessor: ColumnTransformer, fast: bool) -> dict:
    if fast:
        return {
            "random_forest": (
                Pipeline([("preprocess", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
                {"clf__n_estimators": [100], "clf__max_depth": [5]},
            ),
            "xgboost": (
                Pipeline([("preprocess", preprocessor), ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"))]),
                {"clf__n_estimators": [100], "clf__max_depth": [3]},
            ),
        }
    return {
        "random_forest": (
            Pipeline([("preprocess", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 2, 4]},
        ),
        "xgboost": (
            Pipeline([("preprocess", preprocessor), ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"))]),
            {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 4, 5], "clf__learning_rate": [0.01, 0.05, 0.1]},
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Tiny grid + fewer CV folds, for CI smoke runs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"[1/4] Downloading data (fast={args.fast})...")
    df_raw = download_data(args.data_dir)
    df = clean_data(df_raw)
    df.to_csv(os.path.join(args.data_dir, "heart_disease_clean.csv"), index=False)
    print(f"  -> {df.shape[0]} rows after cleaning")

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("[2/4] Training models...")
    cv = StratifiedKFold(n_splits=3 if args.fast else 5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor()
    model_grids = get_model_grids(preprocessor, args.fast)

    results = {}
    for name, (pipeline, param_grid) in model_grids.items():
        gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        results[name] = {"model": best, "metrics": metrics, "best_params": gs.best_params_}
        print(f"  {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    print("[3/4] Selecting best model...")
    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best = results[best_name]
    print(f"  best model: {best_name} (roc_auc={best['metrics']['roc_auc']:.3f})")

    print("[4/4] Saving artifacts...")
    model_path = os.path.join(args.model_dir, "heart_disease_model.joblib")
    joblib.dump(best["model"], model_path)

    metadata = {
        "model_name": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_params": best["best_params"],
        "metrics": best["metrics"],
        "feature_columns": X.columns.tolist(),
        "continuous_cols": CONTINUOUS_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "binary_cols": BINARY_COLS,
        "target_mapping": {"0": "no heart disease", "1": "heart disease present"},
        "fast_mode": args.fast,
    }
    with open(os.path.join(args.model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model -> {model_path}")
    print(f"Saved metadata -> {os.path.join(args.model_dir, 'model_metadata.json')}")


if __name__ == "__main__":
    main()

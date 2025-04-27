#!/usr/bin/env python3
"""cost_sensitive_models.py

Train and evaluate several cost‑sensitive classifiers on one or more CSV files.

Usage (from terminal):
    python cost_sensitive_models.py  dataset_prepared.csv  dataset_smote.csv

If no file names are provided, the script will look for
``dataset_prepared.csv`` and ``dataset_smote.csv`` in the current directory.

The script builds six classifiers, all of them aware of class imbalance:

1. **Weighted Logistic Regression**        – class_weight in loss
2. **Cost‑Sensitive Linear SVM**           – class_weight in hinge loss
3. **Cost‑Sensitive Decision Tree**        – class_weight in Gini entropy
4. **Cost‑Sensitive Random Forest**        – class_weight propagated to every tree
5. **Cost‑Sensitive AdaBoost**             – sample_weight during boosting
6. **Cost‑Sensitive XGBoost (optional)**   – scale_pos_weight, if xgboost is installed

For each data set the script prints and saves a table containing Accuracy,
Balanced Accuracy, Precision, Recall and F1‑score.  All results are also
stored in ``metrics.csv`` for further inspection.

The code runs on Python ≥ 3.8 with scikit‑learn ≥ 1.4.  XGBoost is optional.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Optional – use XGBoost if present
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame | None:
    """Load a CSV file or return *None* if it does not exist."""
    if not os.path.exists(path):
        print(f"[!] File '{path}' not found – skipping.")
        return None
    return pd.read_csv(path)


def prepare_features(
    df: pd.DataFrame, label_col: str | None = None
) -> Tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
    """Split *df* into features *X* and encoded target *y*,
    build a preprocessing transformer that scales numerics and
    one‑hot‑encodes categoricals.
    """

    # Auto‑detect the target column if not specified (last column)
    if label_col is None or label_col not in df.columns:
        label_col = df.columns[-1]

    X = df.drop(columns=[label_col])
    y_raw = df[label_col]

    # Encode labels if they are not numeric
    if not np.issubdtype(y_raw.dtype, np.number):
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw.to_numpy()

    numeric_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    return X, y, preprocessor


def class_weight_dict(y: np.ndarray) -> Dict[int, float]:
    """Return the inverse‑frequency class‑weight mapping used by scikit‑learn."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    k = len(classes)
    return {cls: total / (k * cnt) for cls, cnt in zip(classes, counts)}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return Accuracy, Balanced Accuracy, Precision, Recall, F1 (macro averages)."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, bal_acc, precision, recall, f1

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_models(
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    random_state: int = 42,
):
    """Fit six cost‑sensitive classifiers and return a DataFrame of metrics."""

    # Hold‑out split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Compute class weights once
    cw = class_weight_dict(y_train)

    # Fit preprocessor and transform data once (efficiency!)
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    results: List[Tuple[str, float, float, float, float, float]] = []

    # 1. Weighted Logistic Regression ---------------------------------------
    lr = LogisticRegression(
        max_iter=1000, solver="lbfgs", class_weight=cw, random_state=random_state
    )
    lr.fit(X_train_p, y_train)
    results.append(("WeightedLogReg", *evaluate(y_test, lr.predict(X_test_p))))

    # 2. Cost‑Sensitive Linear SVM -----------------------------------------
    svm = LinearSVC(class_weight=cw, C=1.0, random_state=random_state)
    svm.fit(X_train_p, y_train)
    results.append(("CostSensitiveSVM", *evaluate(y_test, svm.predict(X_test_p))))

    # 3. Decision Tree with class_weight ------------------------------------
    tree = DecisionTreeClassifier(
        class_weight=cw, max_depth=None, random_state=random_state
    )
    tree.fit(X_train_p, y_train)
    results.append(("CostSensitiveTree", *evaluate(y_test, tree.predict(X_test_p))))

    # 4. Random Forest with class_weight ------------------------------------
    forest = RandomForestClassifier(
        n_estimators=300, class_weight=cw, random_state=random_state, n_jobs=-1
    )
    forest.fit(X_train_p, y_train)
    results.append(("CostSensitiveForest", *evaluate(y_test, forest.predict(X_test_p))))

    # 5. AdaBoost with sample_weight ---------------------------------------
    sample_w = np.array([cw[c] for c in y_train])
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
        n_estimators=400,
        learning_rate=0.5,
        random_state=random_state,
    )
    ada.fit(X_train_p, y_train, sample_weight=sample_w)
    results.append(("CostSensitiveAda", *evaluate(y_test, ada.predict(X_test_p))))

    # 6. XGBoost (optional) --------------------------------------------------
    if HAS_XGB:
        if len(np.unique(y_train)) == 2:
            # binary classification – use log‑odds objective
            counts = np.bincount(y_train)
            scale_pos_weight = counts.max() / counts.min()
            xgb = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=-1,
            )
            xgb.fit(X_train_p, y_train)
            y_pred = (xgb.predict(X_test_p) > 0.5).astype(int)
        else:
            # multiclass – softprob + average scale_pos_weight
            classes, counts = np.unique(y_train, return_counts=True)
            spw = float(np.mean(counts.max() / counts))
            xgb = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=len(classes),
                eval_metric="mlogloss",
                scale_pos_weight=spw,
                random_state=random_state,
                n_jobs=-1,
            )
            xgb.fit(X_train_p, y_train)
            y_pred = np.argmax(xgb.predict_proba(X_test_p), axis=1)
        results.append(("CostSensitiveXGB", *evaluate(y_test, y_pred)))

    # ------------------------------------------------------------------
    # Collect metrics in a DataFrame
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "BalancedAccuracy",
            "Precision",
            "Recall",
            "F1",
        ],
    )
    return metrics_df

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main(argv: List[str]) -> None:
    # If no arguments: fall back to default filenames
    dataset_paths = argv or ["dataset_prepared.csv", "dataset_smote.csv"]

    all_metrics = []

    for path in dataset_paths:
        df = load_dataset(path)
        if df is None:
            continue
        print(f"\n=== Processing '{path}' ===")
        X, y, prep = prepare_features(df)
        metrics_df = train_models(X, y, prep)
        metrics_df["Dataset"] = os.path.basename(path)
        all_metrics.append(metrics_df)

    if not all_metrics:
        print("No datasets were processed – nothing to do.")
        return

    combined = pd.concat(all_metrics, ignore_index=True)

    print("\n================ Model Performance ================\n")
    print(combined.to_string(index=False, float_format="%.4f"))

    # Save to CSV for further analysis
    combined.to_csv("metrics.csv", index=False)
    print("\nMetrics written to 'metrics.csv'.")


if __name__ == "__main__":
    main(sys.argv[1:])

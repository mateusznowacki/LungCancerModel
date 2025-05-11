#!/usr/bin/env python3
"""main.py – Train cost-sensitive classifiers and auto-generate visualisations."""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Optional – XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"[!] File '{path}' not found – skipping.")
        return None
    return pd.read_csv(path)

def prepare_features(df: pd.DataFrame, label_col: str | None = None) -> Tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
    if label_col is None or label_col not in df.columns:
        label_col = df.columns[-1]
    X = df.drop(columns=[label_col])
    y_raw = df[label_col]
    y = LabelEncoder().fit_transform(y_raw) if not np.issubdtype(y_raw.dtype, np.number) else y_raw.to_numpy()

    numeric_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return X, y, preprocessor

def class_weight_dict(y: np.ndarray) -> Dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    k = len(classes)
    return {cls: total / (k * cnt) for cls, cnt in zip(classes, counts)}

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float]:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, bal_acc, precision, recall, f1

def ensure_results_folder() -> Path:
    path = Path("results")
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_bar_plot(metrics_df: pd.DataFrame, dataset_name: str, results_dir: Path):
    plt.figure(figsize=(10, 5))
    plt.bar(metrics_df["Model"], metrics_df["F1"], color="skyblue")
    plt.ylabel("F1 Score (macro)")
    plt.title(f"F1 Score by Model — {dataset_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    outfile = results_dir / f"{dataset_name}_f1_bar.png"
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[✓] Bar chart saved: {outfile}")

def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, dataset_name: str, results_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title(f"{model_name} – {dataset_name}")
    plt.tight_layout()
    outfile = results_dir / f"{dataset_name}_{model_name}_cm.png"
    plt.savefig(outfile, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------

def train_models(X: pd.DataFrame, y: np.ndarray, preprocessor: ColumnTransformer, dataset_name: str, random_state: int = 42):
    results: List[Tuple[str, float, float, float, float, float]] = []
    predictions = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    cw = class_weight_dict(y_train)

    # 1. Logistic Regression
    model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight=cw, random_state=random_state)
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)
    results.append(("WeightedLogReg", *evaluate(y_test, y_pred)))
    predictions["WeightedLogReg"] = y_pred

    # 2. Linear SVM
    model = LinearSVC(class_weight=cw, C=1.0, random_state=random_state)
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)
    results.append(("CostSensitiveSVM", *evaluate(y_test, y_pred)))
    predictions["CostSensitiveSVM"] = y_pred

    # 3. Decision Tree
    model = DecisionTreeClassifier(class_weight=cw, random_state=random_state)
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)
    results.append(("CostSensitiveTree", *evaluate(y_test, y_pred)))
    predictions["CostSensitiveTree"] = y_pred

    # 4. Random Forest
    model = RandomForestClassifier(n_estimators=1000, class_weight=cw, random_state=random_state, n_jobs=-1)
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)
    results.append(("CostSensitiveForest", *evaluate(y_test, y_pred)))
    predictions["CostSensitiveForest"] = y_pred

    # 5. AdaBoost
    sample_weight = np.array([cw[c] for c in y_train])
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
        n_estimators=1000,
        learning_rate=0.5,
        random_state=random_state,
    )
    model.fit(X_train_p, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test_p)
    results.append(("CostSensitiveAda", *evaluate(y_test, y_pred)))
    predictions["CostSensitiveAda"] = y_pred

    # 6. XGBoost (optional)
    if HAS_XGB:
        if len(np.unique(y_train)) == 2:
            counts = np.bincount(y_train)
            scale_pos_weight = counts.max() / counts.min()
            model = XGBClassifier(
                n_estimators=1000,
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
            model.fit(X_train_p, y_train)
            y_pred = (model.predict(X_test_p) > 0.5).astype(int)
        else:
            classes, counts = np.unique(y_train, return_counts=True)
            scale_pos_weight = np.mean(counts.max() / counts)
            model = XGBClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=len(classes),
                eval_metric="mlogloss",
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train_p, y_train)
            y_pred = np.argmax(model.predict_proba(X_test_p), axis=1)

        results.append(("CostSensitiveXGB", *evaluate(y_test, y_pred)))
        predictions["CostSensitiveXGB"] = y_pred
        # Extra Trees
        model = ExtraTreesClassifier(n_estimators=1000, class_weight=cw, random_state=random_state, n_jobs=-1)
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        results.append(("CostSensitiveExtraTrees", *evaluate(y_test, y_pred)))
        predictions["CostSensitiveExtraTrees"] = y_pred

        # Gradient Boosting
        model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, random_state=random_state)
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        results.append(("CostSensitiveGBDT", *evaluate(y_test, y_pred)))
        predictions["CostSensitiveGBDT"] = y_pred

    return pd.DataFrame(results, columns=["Model", "Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1"]), y_test, predictions


def save_feature_correlation(X: pd.DataFrame, dataset_name: str, results_dir: Path):
    """Generuje i zapisuje macierz korelacji cech."""
    plt.figure(figsize=(12, 10))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Feature Correlation Matrix – {dataset_name}")
    plt.tight_layout()
    outfile = results_dir / f"{dataset_name}_correlation_matrix.png"
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[✓] Correlation matrix saved: {outfile}")


def save_feature_importance(model, feature_names: List[str], dataset_name: str, model_name: str, results_dir: Path):
    """Generuje i zapisuje wykres ważności cech."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance – {model_name} ({dataset_name})")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        outfile = results_dir / f"{dataset_name}_{model_name}_feature_importance.png"
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"[✓] Feature importance saved: {outfile}")


# ------------------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------------------

def main(argv: List[str]) -> None:
    dataset_paths = argv or ["dataset_prepared.csv", "dataset_smote.csv"]
    results_dir = ensure_results_folder()
    all_metrics = []

    for path in dataset_paths:
        df = load_dataset(path)
        if df is None:
            continue

        print(f"\n=== Processing '{path}' ===")
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        X, y, preproc = prepare_features(df)
        metrics_df, y_test, preds = train_models(X, y, preproc, dataset_name)

        # Save bar-chart and confusion matrices
        save_bar_plot(metrics_df, dataset_name, results_dir)
        for model_name, y_pred in preds.items():
            save_confusion_matrix(y_test, y_pred, model_name, dataset_name, results_dir)

        metrics_df["Dataset"] = dataset_name
        all_metrics.append(metrics_df)

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        print("\n================ Model Performance ================\n")
        print(combined.to_string(index=False, float_format="%.4f"))
        combined.to_csv(results_dir / "metrics.csv", index=False)
        print(f"\nMetrics written to {results_dir/'metrics.csv'}")
    else:
        print("No datasets processed.")

if __name__ == "__main__":
    main(sys.argv[1:])

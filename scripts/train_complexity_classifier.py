#!/usr/bin/env python3
"""
Train Complexity Classifier — compare multiple models, tune the best one.

Loads syscall log data, engineers features, trains 6 classifiers,
compares them via stratified k-fold CV, then hyperparameter-tunes
the winner with proper bias-variance tradeoff analysis.

Usage:
    python scripts/train_complexity_classifier.py
    python scripts/train_complexity_classifier.py --log-path aios/logs/llm_syscalls.jsonl
    python scripts/train_complexity_classifier.py --skip-tuning   # compare only
"""

import argparse
import json
import os
import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_LOG = PROJECT_ROOT / "aios" / "logs" / "llm_syscalls.jsonl"
MODEL_DIR = PROJECT_ROOT / "models"


# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def load_and_prepare(log_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load JSONL, engineer features, return X, y, feature_names."""
    rows = [json.loads(line) for line in open(log_path) if line.strip()]
    df = pd.DataFrame(rows)

    # Derive agent_type
    df["agent_type"] = df["agent_name"].str.replace(r"_r\d+$", "", regex=True)
    df["has_tools"] = df["has_tools"].astype(int)

    # Percentile-based complexity labels (target)
    p33 = df["latency_ms"].quantile(0.33)
    p66 = df["latency_ms"].quantile(0.66)

    def label(ms):
        if ms < p33:
            return "fast"
        if ms < p66:
            return "medium"
        return "large"

    df["complexity"] = df["latency_ms"].apply(label)

    print(f"Loaded {len(df)} rows from {log_path}")
    print(f"Thresholds: fast < {p33:.0f}ms | medium < {p66:.0f}ms | large >= {p66:.0f}ms")
    print(f"Class distribution:\n{df['complexity'].value_counts().to_string()}\n")

    # Feature matrix
    agent_dummies = pd.get_dummies(df["agent_type"], prefix="agent").astype(int)

    # One-hot encode model_name if present and has > 1 unique value
    model_dummies_cols: list[str] = []
    if "model_name" in df.columns and df["model_name"].nunique() > 1:
        model_dummies = pd.get_dummies(df["model_name"], prefix="model").astype(int)
        model_dummies_cols = list(model_dummies.columns)
    else:
        model_dummies = pd.DataFrame()

    # Include message_count (now non-constant with multi-turn)
    base_cols = ["input_char_length", "message_count", "has_tools", "max_tokens", "temperature"]
    feature_cols = base_cols + list(agent_dummies.columns) + model_dummies_cols

    parts = [df[base_cols], agent_dummies]
    if not model_dummies.empty:
        parts.append(model_dummies)
    X = pd.concat(parts, axis=1).values
    y = df["complexity"].values

    print(f"Feature matrix: {X.shape}  ({len(feature_cols)} features)")
    print(f"Features: {feature_cols}\n")

    return df, X, y, feature_cols


# ═══════════════════════════════════════════════════════════════════
# 2. MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════

def get_models() -> dict[str, Pipeline]:
    """Return dict of name -> sklearn Pipeline (scaler + model)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )),
        ]),
    }


def compare_models(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Run stratified 5-fold CV on all models, return comparison DataFrame."""
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    print("=" * 70)
    print("  MODEL COMPARISON — Stratified 5-Fold Cross-Validation")
    print("=" * 70)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "train_accuracy": "accuracy",
    }

    for name, pipeline in models.items():
        cv_results = cross_validate(
            pipeline, X, y, cv=cv,
            scoring={"acc": "accuracy", "f1": "f1_macro"},
            return_train_score=True,
            n_jobs=-1,
        )

        train_acc = cv_results["train_acc"].mean()
        test_acc = cv_results["test_acc"].mean()
        test_f1 = cv_results["test_f1"].mean()
        test_acc_std = cv_results["test_acc"].std()
        test_f1_std = cv_results["test_f1"].std()
        gap = train_acc - test_acc

        # Bias-variance diagnosis
        if test_acc < 0.60:
            diagnosis = "HIGH BIAS (underfitting)"
        elif gap > 0.10:
            diagnosis = "HIGH VARIANCE (overfitting)"
        else:
            diagnosis = "good fit"

        results.append({
            "model": name,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_acc_std": test_acc_std,
            "test_f1": test_f1,
            "test_f1_std": test_f1_std,
            "overfit_gap": gap,
            "diagnosis": diagnosis,
        })

        print(f"\n  {name}")
        print(f"    Train Acc  : {train_acc:.4f}")
        print(f"    Test Acc   : {test_acc:.4f} +/- {test_acc_std:.4f}")
        print(f"    Test F1    : {test_f1:.4f} +/- {test_f1_std:.4f}")
        print(f"    Overfit Gap: {gap:.4f}  -> {diagnosis}")

    df_results = pd.DataFrame(results).sort_values("test_f1", ascending=False)

    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    x_pos = range(len(df_results))
    axes[0].barh(
        df_results["model"], df_results["test_acc"],
        xerr=df_results["test_acc_std"], color="steelblue", edgecolor="white",
    )
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Test Accuracy (5-Fold CV)")
    axes[0].set_xlim(0, 1.05)

    axes[1].barh(
        df_results["model"], df_results["test_f1"],
        xerr=df_results["test_f1_std"], color="teal", edgecolor="white",
    )
    axes[1].set_xlabel("F1 Macro")
    axes[1].set_title("Test F1-Macro (5-Fold CV)")
    axes[1].set_xlim(0, 1.05)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "model_comparison.png", dpi=150)
    plt.show()

    # Bias-variance plot (train vs test)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df_results))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df_results["train_acc"].values,
           width, label="Train Acc", color="#3498db", edgecolor="white")
    ax.bar([i + width / 2 for i in x], df_results["test_acc"].values,
           width, label="Test Acc", color="#e74c3c", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_results["model"].values, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Bias-Variance: Train vs Test Accuracy")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "bias_variance_comparison.png", dpi=150)
    plt.show()

    print("\n" + "=" * 70)
    best = df_results.iloc[0]
    print(f"  BEST MODEL: {best['model']}")
    print(f"    Test F1 Macro: {best['test_f1']:.4f}")
    print(f"    Overfit Gap  : {best['overfit_gap']:.4f} ({best['diagnosis']})")
    print("=" * 70)

    return df_results


# ═══════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════

PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["saga"],
    },
    "KNN (k=5)": {
        "clf__n_neighbors": [3, 5, 7, 11, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
    "SVM (RBF)": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto", 0.01, 0.1],
        "clf__kernel": ["rbf", "poly"],
    },
    "Decision Tree": {
        "clf__max_depth": [3, 5, 7, 10, 15, None],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__max_depth": [3, 5, 7, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2", None],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__min_samples_leaf": [1, 2, 5],
    },
}


def tune_best_model(
    best_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list[str],
) -> Pipeline:
    """GridSearchCV on the best model, return tuned pipeline."""
    models = get_models()
    pipeline = models[best_name]
    param_grid = PARAM_GRIDS[best_name]

    print("\n" + "=" * 70)
    print(f"  HYPERPARAMETER TUNING: {best_name}")
    print(f"  Grid size: {param_grid}")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    print(f"\n  Best params: {search.best_params_}")
    print(f"  Best CV F1 : {search.best_score_:.4f}")

    # Train vs CV score for bias-variance check
    cv_results = pd.DataFrame(search.cv_results_)
    best_idx = search.best_index_
    train_score = cv_results.loc[best_idx, "mean_train_score"]
    test_score = cv_results.loc[best_idx, "mean_test_score"]
    print(f"  Train F1   : {train_score:.4f}")
    print(f"  CV F1      : {test_score:.4f}")
    print(f"  Gap        : {train_score - test_score:.4f}")

    # Evaluate on held-out test set
    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    print(f"\n  HELD-OUT TEST SET RESULTS:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_test, y_pred, labels=["fast", "medium", "large"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["fast", "medium", "large"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {best_name} (tuned)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.show()

    return best_pipeline


# ═══════════════════════════════════════════════════════════════════
# 4. LEARNING CURVES (BIAS-VARIANCE ANALYSIS)
# ═══════════════════════════════════════════════════════════════════

def plot_learning_curves(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray, title: str
):
    """Plot learning curves to diagnose bias vs variance."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="f1_macro",
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color="#3498db")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color="#e74c3c")
    ax.plot(train_sizes, train_mean, "o-", color="#3498db", label="Training F1")
    ax.plot(train_sizes, test_mean, "o-", color="#e74c3c", label="Validation F1")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Macro Score")
    ax.set_title(f"Learning Curves — {title}")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotate final gap
    final_gap = train_mean[-1] - test_mean[-1]
    ax.annotate(
        f"Final gap: {final_gap:.3f}",
        xy=(train_sizes[-1], test_mean[-1]),
        xytext=(train_sizes[-1] * 0.7, test_mean[-1] - 0.12),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10, color="gray",
    )

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "learning_curves.png", dpi=150)
    plt.show()

    print(f"Learning curve summary:")
    print(f"  Train F1 (final) : {train_mean[-1]:.4f}")
    print(f"  Val F1   (final) : {test_mean[-1]:.4f}")
    print(f"  Gap              : {final_gap:.4f}")
    if final_gap > 0.10:
        print("  Diagnosis: HIGH VARIANCE — consider regularization or more data")
    elif test_mean[-1] < 0.60:
        print("  Diagnosis: HIGH BIAS — consider more features or complex model")
    else:
        print("  Diagnosis: Good bias-variance tradeoff")


# ═══════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def plot_feature_importance(pipeline: Pipeline, feature_names: list[str]):
    """Plot feature importance for tree-based models."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        print("  (model does not expose feature_importances_, skipping)")
        return

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="steelblue", edgecolor="white",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (tuned model)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importances.png", dpi=150)
    plt.show()

    print("\nFeature importances:")
    for i in indices:
        print(f"  {feature_names[i]:<35s} {importances[i]:.4f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train complexity classifier")
    parser.add_argument(
        "--log-path", type=str, default=str(DEFAULT_LOG),
        help="Path to llm_syscalls.jsonl",
    )
    parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Only compare models, skip hyperparameter tuning",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Held-out test set fraction (default: 0.2)",
    )
    args = parser.parse_args()

    # Create output directory
    MODEL_DIR.mkdir(exist_ok=True)

    # ── Load data ───────────────────────────────────────────────
    df, X, y, feature_names = load_and_prepare(Path(args.log_path))

    # ── Train/test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")

    # ── Compare all models ──────────────────────────────────────
    results_df = compare_models(X_train, y_train, feature_names)
    results_df.to_csv(MODEL_DIR / "comparison_results.csv", index=False)

    best_name = results_df.iloc[0]["model"]

    if args.skip_tuning:
        print("\n  --skip-tuning set, stopping after comparison.")
        return

    # ── Hyperparameter tune the best ────────────────────────────
    best_pipeline = tune_best_model(
        best_name, X_train, y_train, X_test, y_test, feature_names
    )

    # ── Learning curves ─────────────────────────────────────────
    plot_learning_curves(best_pipeline, X, y, f"{best_name} (tuned)")

    # ── Feature importance ──────────────────────────────────────
    plot_feature_importance(best_pipeline, feature_names)

    # ── Save model ──────────────────────────────────────────────
    model_path = MODEL_DIR / "complexity_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "pipeline": best_pipeline,
            "feature_names": feature_names,
            "best_model_name": best_name,
            "best_params": best_pipeline.get_params(),
        }, f)
    print(f"\n  Model saved to {model_path}")

    # ── Final summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Best model        : {best_name}")
    print(f"  Features          : {len(feature_names)}")
    print(f"  Train size        : {X_train.shape[0]}")
    print(f"  Test size         : {X_test.shape[0]}")
    print(f"  Artifacts saved to: {MODEL_DIR}/")
    print(f"    - complexity_classifier.pkl")
    print(f"    - comparison_results.csv")
    print(f"    - model_comparison.png")
    print(f"    - bias_variance_comparison.png")
    print(f"    - confusion_matrix.png")
    print(f"    - learning_curves.png")
    print(f"    - feature_importances.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

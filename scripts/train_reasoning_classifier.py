#!/usr/bin/env python3
"""
Train 4-Class Request Type Classifier for LLM Agent Scheduling.

Classifies LLM agent requests into 4 reasoning patterns:
  - Simple QA:        short output, single-turn, no tools
  - Conversational:   multi-turn dialogue, no tools
  - Tool-Augmented:   agentic reasoning with tool calls
  - Long Generation:  extended output, single-turn, no tools

Labels are defined from (max_tokens, message_count, has_tools).
Classifier features EXCLUDE max_tokens and has_tools directly,
forcing the model to learn from proxy signals (input_char_length,
temperature, agent_type, model_name, message_count).

Split: 60/20/20 train/val/test.

Usage:
    source .venv/bin/activate
    python scripts/train_reasoning_classifier.py
"""

import argparse
import json
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
from sklearn.preprocessing import StandardScaler
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

CLASS_ORDER = ["Simple QA", "Conversational", "Tool-Augmented", "Long Generation"]


# ═══════════════════════════════════════════════════════════════════
# 1. LABELING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def assign_request_type(row) -> str:
    mt = row["max_tokens"] or 512
    mc = row["message_count"] or 2
    tools = bool(row["has_tools"])

    if tools:
        return "Tool-Augmented"
    if mc > 2:
        return "Conversational"
    if mt > 512:
        return "Long Generation"
    return "Simple QA"


# ═══════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_and_prepare(log_paths: list) -> tuple:
    all_rows = []
    for lp in log_paths:
        rows = [json.loads(line) for line in open(lp) if line.strip()]
        all_rows.extend(rows)
        print(f"  Loaded {len(rows)} rows from {lp}")

    df = pd.DataFrame(all_rows)
    if "model_name" in df.columns:
        df = df[df["model_name"].notna()]

    df["agent_type"] = df["agent_name"].str.replace(r"_r\d+$", "", regex=True)
    df["has_tools"] = df["has_tools"].astype(int)
    df["request_type"] = df.apply(assign_request_type, axis=1)

    print(f"\nLoaded {len(df)} rows total")
    print(f"\nClass distribution:")
    for cls in CLASS_ORDER:
        count = (df["request_type"] == cls).sum()
        print(f"  {cls:<20s}: {count:>5d}  ({count/len(df)*100:.1f}%)")

    # Features: EXCLUDE max_tokens and has_tools (used in labeling)
    agent_dummies = pd.get_dummies(df["agent_type"], prefix="agent").astype(int)

    model_dummies_cols = []
    if "model_name" in df.columns and df["model_name"].nunique() > 1:
        model_dummies = pd.get_dummies(df["model_name"], prefix="model").astype(int)
        model_dummies_cols = list(model_dummies.columns)
    else:
        model_dummies = pd.DataFrame()

    base_cols = ["input_char_length", "message_count", "temperature"]
    feature_cols = base_cols + list(agent_dummies.columns) + model_dummies_cols

    parts = [df[base_cols], agent_dummies]
    if not model_dummies.empty:
        parts.append(model_dummies)
    X = pd.concat(parts, axis=1).values
    y = df["request_type"].values

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"  NOTE: max_tokens and has_tools excluded from features\n")

    return df, X, y, feature_cols


# ═══════════════════════════════════════════════════════════════════
# 3. MODELS
# ═══════════════════════════════════════════════════════════════════

def get_models() -> dict:
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
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════

def compare_models(X_train, y_train, X_val, y_val, feature_names) -> pd.DataFrame:
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("=" * 70)
    print("  MODEL COMPARISON — 4-Class Request Type Classifier")
    print("=" * 70)

    for name, pipeline in models.items():
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"acc": "accuracy", "f1": "f1_macro"},
            return_train_score=True, n_jobs=-1,
        )
        cv_train_acc = cv_results["train_acc"].mean()
        cv_f1 = cv_results["test_f1"].mean()
        cv_f1_std = cv_results["test_f1"].std()
        gap = cv_train_acc - cv_results["test_acc"].mean()

        pipeline.fit(X_train, y_train)
        y_vp = pipeline.predict(X_val)
        val_f1 = f1_score(y_val, y_vp, average="macro")
        val_acc = (y_vp == y_val).mean()

        if val_f1 < 0.40:
            diag = "HIGH BIAS"
        elif gap > 0.15:
            diag = "HIGH VARIANCE"
        else:
            diag = "good fit"

        results.append({"model": name, "cv_train_acc": cv_train_acc,
                         "cv_f1": cv_f1, "cv_f1_std": cv_f1_std,
                         "val_f1": val_f1, "val_acc": val_acc,
                         "gap": gap, "diagnosis": diag})

        print(f"\n  {name}")
        print(f"    CV Train Acc : {cv_train_acc:.4f}")
        print(f"    CV F1        : {cv_f1:.4f} +/- {cv_f1_std:.4f}")
        print(f"    Val F1       : {val_f1:.4f}   Val Acc: {val_acc:.4f}")
        print(f"    Gap          : {gap:.4f}  -> {diag}")

    df_r = pd.DataFrame(results).sort_values("val_f1", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].barh(df_r["model"], df_r["cv_f1"], xerr=df_r["cv_f1_std"],
                 color="steelblue", edgecolor="white")
    axes[0].set_xlabel("F1 Macro"); axes[0].set_title("CV F1 (5-Fold on Train)")
    axes[0].set_xlim(0, 1.05)
    axes[1].barh(df_r["model"], df_r["val_f1"], color="teal", edgecolor="white")
    axes[1].set_xlabel("F1 Macro"); axes[1].set_title("Validation Set F1")
    axes[1].set_xlim(0, 1.05)
    plt.tight_layout(); plt.savefig(MODEL_DIR / "reasoning_model_comparison.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df_r)); w = 0.35
    ax.bar([i-w/2 for i in x], df_r["cv_train_acc"].values, w, label="Train", color="#3498db")
    ax.bar([i+w/2 for i in x], df_r["val_acc"].values, w, label="Val", color="#e74c3c")
    ax.set_xticks(list(x)); ax.set_xticklabels(df_r["model"].values, rotation=25, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_title("Train vs Val Accuracy"); ax.legend(); ax.set_ylim(0,1.05)
    plt.tight_layout(); plt.savefig(MODEL_DIR / "reasoning_bias_variance.png", dpi=150); plt.close()

    best = df_r.iloc[0]
    print(f"\n{'='*70}\n  BEST: {best['model']}  Val F1={best['val_f1']:.4f}\n{'='*70}")
    return df_r


# ═══════════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════

PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2"], "clf__solver": ["saga"],
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


def tune_best_model(best_name, X_train, y_train, X_val, y_val):
    models = get_models()
    pipeline = models[best_name]

    print(f"\n{'='*70}\n  TUNING: {best_name}\n{'='*70}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, PARAM_GRIDS[best_name], cv=cv,
                          scoring="f1_macro", n_jobs=-1, verbose=1,
                          return_train_score=True)
    search.fit(X_train, y_train)

    cv_df = pd.DataFrame(search.cv_results_)
    bi = search.best_index_
    print(f"\n  Best params : {search.best_params_}")
    print(f"  CV Train F1 : {cv_df.loc[bi, 'mean_train_score']:.4f}")
    print(f"  CV Val F1   : {search.best_score_:.4f}")
    print(f"  Gap         : {cv_df.loc[bi, 'mean_train_score'] - search.best_score_:.4f}")

    bp = search.best_estimator_
    y_vp = bp.predict(X_val)
    print(f"\n  VALIDATION (tuned): F1={f1_score(y_val, y_vp, average='macro'):.4f}")
    print(classification_report(y_val, y_vp, labels=CLASS_ORDER, digits=4))
    return bp


def final_test(pipeline, X_test, y_test, best_name):
    print(f"\n{'='*70}\n  FINAL TEST SET (held-out)\n{'='*70}")
    y_p = pipeline.predict(X_test)
    tf1 = f1_score(y_test, y_p, average="macro")
    print(f"\n  Test F1 Macro: {tf1:.4f}")
    print(classification_report(y_test, y_p, labels=CLASS_ORDER, digits=4))

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_p, labels=CLASS_ORDER)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Test Confusion Matrix — {best_name} (tuned)")
    plt.tight_layout(); plt.savefig(MODEL_DIR / "reasoning_confusion_matrix.png", dpi=150); plt.close()
    return tf1


# ═══════════════════════════════════════════════════════════════════
# 6. PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_learning_curves(pipeline, X, y, title):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ts, tr_s, te_s = learning_curve(pipeline, X, y, cv=cv, n_jobs=-1,
                                     train_sizes=np.linspace(0.1, 1.0, 10),
                                     scoring="f1_macro", random_state=42)
    tr_m, te_m = tr_s.mean(axis=1), te_s.mean(axis=1)
    tr_std, te_std = tr_s.std(axis=1), te_s.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(ts, tr_m-tr_std, tr_m+tr_std, alpha=0.1, color="#3498db")
    ax.fill_between(ts, te_m-te_std, te_m+te_std, alpha=0.1, color="#e74c3c")
    ax.plot(ts, tr_m, "o-", color="#3498db", label="Train F1")
    ax.plot(ts, te_m, "o-", color="#e74c3c", label="Val F1")
    ax.set_xlabel("Training Size"); ax.set_ylabel("F1 Macro")
    ax.set_title(f"Learning Curves — {title}")
    ax.legend(loc="lower right"); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
    g = tr_m[-1] - te_m[-1]
    ax.annotate(f"Gap: {g:.3f}", xy=(ts[-1], te_m[-1]),
                xytext=(ts[-1]*0.7, te_m[-1]-0.1),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=10, color="gray")
    plt.tight_layout(); plt.savefig(MODEL_DIR / "reasoning_learning_curves.png", dpi=150); plt.close()
    print(f"\nLearning curves: Train F1={tr_m[-1]:.4f}, Val F1={te_m[-1]:.4f}, Gap={g:.4f}")


def plot_feature_importance(pipeline, feature_names):
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        print("  (no feature_importances_, skipping)")
        return
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([feature_names[i] for i in idx], imp[idx], color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance"); ax.set_title("Feature Importances — Request Type Classifier")
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(MODEL_DIR / "reasoning_feature_importances.png", dpi=150); plt.close()
    print("\nFeature importances:")
    for i in idx:
        print(f"  {feature_names[i]:<35s} {imp[i]:.4f}")


def plot_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df["request_type"].value_counts().reindex(CLASS_ORDER)
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    axes[0].bar(CLASS_ORDER, counts.values, color=colors, edgecolor="white")
    axes[0].set_ylabel("Count"); axes[0].set_title("Request Type Distribution")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v+10, str(v), ha="center", fontweight="bold")

    class_data = [df[df["request_type"]==c]["latency_ms"].values for c in CLASS_ORDER]
    bp = axes[1].boxplot(class_data, labels=CLASS_ORDER, patch_artist=True)
    for p, c in zip(bp["boxes"], colors):
        p.set_facecolor(c); p.set_alpha(0.7)
    axes[1].set_ylabel("Latency (ms)"); axes[1].set_title("Latency by Request Type")
    plt.tight_layout(); plt.savefig(MODEL_DIR / "reasoning_class_distribution.png", dpi=150); plt.close()

    print("\nLatency by request type:")
    for cls in CLASS_ORDER:
        s = df[df["request_type"]==cls]["latency_ms"]
        print(f"  {cls:<20s}: median={s.median():.0f}ms, mean={s.mean():.0f}ms, p95={s.quantile(0.95):.0f}ms")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-paths", type=str, nargs="+", default=[str(DEFAULT_LOG)])
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)

    df, X, y, feat_names = load_and_prepare([Path(p) for p in args.log_paths])
    plot_class_distribution(df)

    # 60/20/20 split
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)
    print(f"\nSplit: Train={X_train.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}  (60/20/20)\n")

    results_df = compare_models(X_train, y_train, X_val, y_val, feat_names)
    results_df.to_csv(MODEL_DIR / "reasoning_comparison_results.csv", index=False)
    best_name = results_df.iloc[0]["model"]

    if args.skip_tuning:
        print("\n  Stopped after comparison (--skip-tuning).")
        return

    bp = tune_best_model(best_name, X_train, y_train, X_val, y_val)
    plot_learning_curves(bp, X_train, y_train, f"{best_name} (tuned)")
    plot_feature_importance(bp, feat_names)
    tf1 = final_test(bp, X_test, y_test, best_name)

    with open(MODEL_DIR / "reasoning_classifier.pkl", "wb") as f:
        pickle.dump({"pipeline": bp, "feature_names": feat_names,
                      "best_model_name": best_name, "class_order": CLASS_ORDER,
                      "test_f1": tf1,
                      "taxonomy": {"Simple QA": "max_tokens<=128, mc<=2, no tools",
                                   "Conversational": "mc>2, no tools",
                                   "Tool-Augmented": "has_tools=True",
                                   "Long Generation": "max_tokens>512, mc<=2, no tools"}}, f)

    print(f"\n{'='*70}")
    print(f"  DONE — {best_name}, Test F1={tf1:.4f}")
    print(f"  Train/Val/Test: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
    print(f"  Saved: models/reasoning_classifier.pkl + 6 PNGs")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

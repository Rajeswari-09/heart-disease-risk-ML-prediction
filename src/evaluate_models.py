"""
evaluate_models.py
------------------
Evaluates all trained pipelines on the held-out test set.
Generates:
  - Confusion matrices for every model
  - Combined ROC curve plot
  - Model comparison table (reports/model_results.csv)

Run standalone:
    python src/evaluate_models.py

Or import and call evaluate_all_models() from the notebook.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

from data_inspection import load_data
from preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, build_full_pipeline
from train_models import (
    prepare_data,
    get_classifiers,
    tune_best_models,
    run_cross_validation,
    RANDOM_STATE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Consistent colour palette
PALETTE = sns.color_palette("tab10")


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------
def evaluate_pipeline(name: str, pipeline, X_test, y_test) -> dict:
    """Compute all classification metrics for one fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(name: str, y_test, y_pred, save: bool = True) -> None:
    """Plot and save a confusion matrix."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        fname = FIGURES_DIR / f"cm_{name.lower().replace(' ', '_')}.png"
        fig.savefig(fname, dpi=150)
        print(f"  [SAVED] {fname.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC curves (all models on one plot)
# ---------------------------------------------------------------------------
def plot_roc_curves(pipelines: dict, X_test, y_test, save: bool = True) -> None:
    """Plot ROC curves for all models on a single figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.50)")

    for idx, (name, pipe) in enumerate(pipelines.items()):
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=PALETTE[idx % len(PALETTE)],
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fname = FIGURES_DIR / "roc_curves_all_models.png"
        fig.savefig(fname, dpi=150)
        print(f"  [SAVED] {fname.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison(results_df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart comparing all models across key metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    plot_df = results_df.set_index("Model")[metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df))
    width = 0.14

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, plot_df[metric], width, label=metric,
                      color=PALETTE[i])

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(plot_df.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Test Set Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fname = FIGURES_DIR / "model_comparison.png"
        fig.savefig(fname, dpi=150)
        print(f"  [SAVED] {fname.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_all_models(
    pipelines: dict,
    X_test,
    y_test,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all pipelines on the test set.
    Returns a DataFrame with all metrics.

    NOTE: This function works with ANY dict of {name: fitted_pipeline}.
    It can receive either untuned or tuned pipelines.
    """
    all_metrics = []

    for name, pipeline in pipelines.items():
        metrics, y_pred, y_prob = evaluate_pipeline(name, pipeline, X_test, y_test)
        all_metrics.append(metrics)

        if verbose:
            print(f"\n[EVAL] {name}")
            print(classification_report(y_test, y_pred,
                                        target_names=["No Disease", "Disease"],
                                        digits=4))

        # Save confusion matrix
        plot_confusion_matrix(name, y_test, y_pred)

    # ROC curves (all on one plot)
    plot_roc_curves(pipelines, X_test, y_test)

    results_df = pd.DataFrame(all_metrics).sort_values("ROC_AUC", ascending=False)
    results_df = results_df.round(4)

    # Bar chart
    plot_model_comparison(results_df)

    # Save results CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Model results saved to: {results_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON TABLE — TEST SET")
    print("=" * 80)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(results_df.to_string(index=False))

    print("\n[IMPORTANT] Medical Classification Note:")
    print("""
  In heart disease prediction, FALSE NEGATIVES are critical:
  - A false negative means a patient WITH disease is predicted as HEALTHY.
  - This could delay treatment and have serious health consequences.
  - Therefore, RECALL (Sensitivity) deserves special attention alongside AUC.
  - A model with slightly lower accuracy but higher recall may be preferable
    in clinical screening contexts.
  - This model is NOT a medical diagnostic tool. It is for research only.
""")

    return results_df


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  HEART DISEASE PREDICTION — MODEL EVALUATION")
    print("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    classifiers = get_classifiers()

    print("\n[STEP 1] Running cross-validation ...")
    cv_df = run_cross_validation(classifiers, X_train, y_train)

    print("\n[STEP 2] Tuning models ...")
    tuned_pipelines = tune_best_models(classifiers, X_train, y_train)

    # Add un-tuned models for evaluation comparison
    all_pipelines = {}
    for name, clf in classifiers.items():
        pipe = build_full_pipeline(clf, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        pipe.fit(X_train, y_train)
        all_pipelines[f"{name} (base)"] = pipe

    for name, pipe in tuned_pipelines.items():
        all_pipelines[f"{name} (tuned)"] = pipe

    print("\n[STEP 3] Evaluating all models on test set ...")
    results_df = evaluate_all_models(all_pipelines, X_test, y_test)
    return results_df


if __name__ == "__main__":
    main()

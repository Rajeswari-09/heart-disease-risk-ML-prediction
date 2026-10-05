"""
explain_model.py
----------------
Model interpretation for the saved heart disease prediction model.

Generates:
  1. Feature importance (tree-based models — Random Forest / XGBoost)
  2. Permutation importance (model-agnostic)
  3. SHAP summary plot (TreeExplainer for tree-based models)

All figures are saved to reports/figures/.

Run:
    python src/explain_model.py

IMPORTANT:
  Feature importance and SHAP values show which features are most
  associated with the model's predictions. They do NOT establish
  causal relationships between features and heart disease.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_inspection import load_data
from preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from train_models import prepare_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(path: Path = MODELS_PATH):
    """Load the saved pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_models.py first."
        )
    pipeline = joblib.load(path)
    print(f"[INFO] Model loaded from: {path}")
    return pipeline


# ---------------------------------------------------------------------------
# Helper: get transformed feature names
# ---------------------------------------------------------------------------
def get_feature_names(pipeline) -> list:
    """Extract feature names from the fitted ColumnTransformer."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# Tree-based feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(pipeline, feature_names: list, top_n: int = 20) -> None:
    """
    Extract and plot built-in feature importance from tree-based models.
    Available for: Random Forest, Decision Tree, Gradient Boosting, XGBoost.
    """
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        print("[SKIP] Classifier does not support feature_importances_. Skipping.")
        return

    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    # Shorten feature names for readability
    fi_df["Feature"] = fi_df["Feature"].str.replace("num__", "").str.replace("cat__", "")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=fi_df, y="Feature", x="Importance", palette="viridis", ax=ax)
    ax.set_title(
        f"Feature Importance (Top {top_n}) — {type(clf).__name__}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Importance Score (Mean Decrease in Impurity)", fontsize=11)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = FIGURES_DIR / "feature_importance_tree.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {fname.name}")


# ---------------------------------------------------------------------------
# Permutation importance (model-agnostic)
# ---------------------------------------------------------------------------
def plot_permutation_importance(
    pipeline, X_test, y_test, feature_names: list, top_n: int = 20
) -> None:
    """
    Compute permutation importance on the test set.
    Model-agnostic: works for any classifier.
    Shows by how much model performance drops when a feature is shuffled.
    """
    from sklearn.inspection import permutation_importance

    print("[INFO] Computing permutation importance (may take ~30s) ...")
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=20,
        random_state=42,
        scoring="roc_auc",
        n_jobs=1,
    )

    # Map importances back to original feature names (X_test columns)
    all_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    pi_df = pd.DataFrame({
        "Feature": all_cols[:len(result.importances_mean)],
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        pi_df["Feature"][::-1],
        pi_df["Importance_Mean"][::-1],
        xerr=pi_df["Importance_Std"][::-1],
        color="steelblue",
        ecolor="black",
        capsize=4,
    )
    ax.set_xlabel("Mean Decrease in ROC-AUC when feature is permuted", fontsize=11)
    ax.set_title(
        f"Permutation Importance (Top {top_n}) — Test Set",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = FIGURES_DIR / "permutation_importance.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {fname.name}")

    print("\nTop features by permutation importance:")
    print(pi_df.to_string(index=False))


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------
def plot_shap_summary(pipeline, X_train, feature_names: list) -> None:
    """
    Generate a SHAP beeswarm summary plot.
    - TreeExplainer for tree-based models (RF, GBM, XGBoost, DT)
    - LinearExplainer for Logistic Regression / LinearSVM
    - Gracefully skips for SVC (kernel) and KNN which SHAP does not natively support.

    SHAP values show how each feature pushes the model output higher or lower
    relative to the base rate. They are interpretable but do NOT imply causation.
    """
    try:
        import shap
    except ImportError:
        print("[SKIP] shap not installed. Run: pip install shap")
        return

    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Transform training data
    X_transformed = preprocessor.transform(X_train)

    # Shorten feature names
    short_names = [
        fn.replace("num__", "").replace("cat__", "")
        for fn in feature_names
    ]

    print("[INFO] Computing SHAP values ...")
    shap_vals = None

    # Try TreeExplainer
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values if not isinstance(shap_values, list) else shap_values[0]
        print("[INFO] Using TreeExplainer.")
    except Exception:
        pass

    # Try LinearExplainer (Logistic Regression)
    if shap_vals is None:
        try:
            background = shap.maskers.Independent(X_transformed, max_samples=200)
            explainer = shap.LinearExplainer(clf, background)
            shap_vals = explainer.shap_values(X_transformed)
            print("[INFO] Using LinearExplainer.")
        except Exception:
            pass

    if shap_vals is None:
        print("[SKIP] SHAP not supported for this model type (e.g., SVC with RBF kernel).")
        print("[INFO] Use permutation importance results instead (already saved).")
        return

    fig, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_transformed,
        feature_names=short_names,
        show=False,
        plot_size=(10, 8),
    )
    plt.title("SHAP Summary Plot -- Feature Impact on Prediction",
              fontsize=13, fontweight="bold")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = FIGURES_DIR / "shap_summary.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {fname.name}")

    print("""
[NOTE] SHAP Interpretation:
  - Each point represents one patient.
  - Color = feature value (red=high, blue=low).
  - X-axis = SHAP value = impact on model output (positive -> pushes toward disease).
  - Features are ranked by mean absolute SHAP value (most important at top).
  - SHAP shows model associations, NOT causal relationships.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  HEART DISEASE PREDICTION — EXPLAINABILITY")
    print("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    pipeline = load_model()
    feature_names = get_feature_names(pipeline)

    print(f"\n[INFO] Transformed features ({len(feature_names)}):")
    for fn in feature_names:
        print(f"  {fn}")

    print("\n[STEP 1] Tree-based feature importance")
    plot_feature_importance(pipeline, feature_names)

    print("\n[STEP 2] Permutation importance")
    plot_permutation_importance(pipeline, X_test, y_test, feature_names)

    print("\n[STEP 3] SHAP analysis")
    plot_shap_summary(pipeline, X_train, feature_names)

    print("\n[DONE] Explainability plots saved to reports/figures/")
    print("\n[IMPORTANT] Disclaimer:")
    print("  Feature importance values describe model associations only.")
    print("  They do NOT prove that a feature causes heart disease.")
    print("  Do not interpret these results as medical advice.")


if __name__ == "__main__":
    main()

"""
train_models.py
---------------
Loads data, builds full sklearn pipelines for multiple classifiers,
performs Stratified K-Fold cross-validation, runs GridSearchCV on
selected models, saves the best model, and writes model_results.csv.

Run:
    python src/train_models.py
"""

import sys
from pathlib import Path

# Allow importing sibling modules when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_inspection import load_data, get_feature_lists
from preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    build_full_pipeline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20        # 80/20 split
CV_FOLDS = 5            # Stratified K-Fold
SCORING = "roc_auc"     # Primary CV metric (appropriate for medical classification)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    df = df.copy()

    # Treat impossible zero values as missing
    for col in ["chol", "trestbps"]:
        df[col] = df[col].replace(0, np.nan)
    """
    Split into X, y (binary target) and perform train/test split.

    Stratification ensures that the proportion of positive/negative cases
    is preserved in both training and test sets, which is critical when
    dealing with class imbalance in medical datasets.
    """
    # Binary target: 0 = no disease, 1 = disease present
    y = (df["num"] > 0).astype(int)

    # Drop non-predictive identifiers and the raw target
    drop_cols = ["id", "dataset", "num"]
    X = df.drop(columns=drop_cols)

    # Cast boolean-typed object columns to string so OHE handles them cleanly
    bool_cols = X.select_dtypes(include="object").columns[
        X.select_dtypes(include="object").apply(
            lambda c: c.dropna().isin(["True", "False"]).all()
        )
    ].tolist()
    for col in bool_cols:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Preserves class ratio in each split
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"[INFO] Train positive rate: {y_train.mean():.3f} | Test positive rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def get_classifiers() -> dict:
    """Return a dict of {name: unfitted estimator}."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Support Vector Machine": SVC(
            probability=True, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def run_cross_validation(classifiers: dict, X_train, y_train) -> pd.DataFrame:
    """
    Run Stratified K-Fold CV on all classifiers.
    Reports mean and std of ROC-AUC and Accuracy across folds.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    for name, clf in classifiers.items():
        print(f"  [CV] {name} ...", end=" ", flush=True)
        pipeline = build_full_pipeline(clf, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=skf,
            scoring=["roc_auc", "accuracy", "f1", "precision", "recall"],
            n_jobs=1,
        )
        row = {
            "Model": name,
            "CV_ROC_AUC_Mean": scores["test_roc_auc"].mean(),
            "CV_ROC_AUC_Std": scores["test_roc_auc"].std(),
            "CV_Accuracy_Mean": scores["test_accuracy"].mean(),
            "CV_Accuracy_Std": scores["test_accuracy"].std(),
            "CV_F1_Mean": scores["test_f1"].mean(),
            "CV_Recall_Mean": scores["test_recall"].mean(),
            "CV_Precision_Mean": scores["test_precision"].mean(),
        }
        cv_results.append(row)
        print(f"ROC-AUC={row['CV_ROC_AUC_Mean']:.4f} ± {row['CV_ROC_AUC_Std']:.4f}")

    return pd.DataFrame(cv_results).sort_values("CV_ROC_AUC_Mean", ascending=False)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_best_models(classifiers: dict, X_train, y_train) -> dict:
    """
    Run GridSearchCV on selected models.

    The test set is never used during tuning.
    The best tuned CV ROC-AUC score is stored on each fitted pipeline
    so the final model can be selected using training-only CV results.
    """
    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    tuned = {}
    tuning_rows = []

    param_grids = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
        },
        "Support Vector Machine": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["rbf", "linear"],
        },
    }

    for name, param_grid in param_grids.items():
        if name not in classifiers:
            continue

        print(f"  [TUNE] {name} ...", end=" ", flush=True)

        pipeline = build_full_pipeline(
            classifiers[name],
            NUMERICAL_FEATURES,
            CATEGORICAL_FEATURES
        )

        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring=SCORING,
            n_jobs=1,
            refit=True
        )

        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_

        # Store the tuned CV score directly on the fitted pipeline
        best_pipeline._tuned_cv_roc_auc = search.best_score_
        best_pipeline._tuned_best_params = search.best_params_

        tuned[name] = best_pipeline

        tuning_rows.append({
            "Model": name,
            "Tuned_CV_ROC_AUC": round(search.best_score_, 4),
            "Best_Params": str(search.best_params_)
        })

        print(
            f"Best ROC-AUC={search.best_score_:.4f} | "
            f"Params={search.best_params_}"
        )

    # Save tuning results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        "Tuned_CV_ROC_AUC",
        ascending=False
    )
    tuning_df.to_csv(
        REPORTS_DIR / "tuning_results.csv",
        index=False
    )

    return tuned


def save_best_model(tuned_pipelines: dict):
    """
    Select and save the tuned model with the highest
    training-only cross-validation ROC-AUC.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not tuned_pipelines:
        raise ValueError("No tuned pipelines available.")

    best_name, best_pipeline = max(
        tuned_pipelines.items(),
        key=lambda item: item[1]._tuned_cv_roc_auc
    )

    save_path = MODELS_DIR / "heart_disease_model.pkl"
    joblib.dump(best_pipeline, save_path)

    print(
        f"\n[INFO] Final model selected using tuned CV ROC-AUC: "
        f"'{best_name}'"
    )
    print(
        f"[INFO] Tuned CV ROC-AUC: "
        f"{best_pipeline._tuned_cv_roc_auc:.4f}"
    )
    print(f"[INFO] Saved to: {save_path}")

    return best_name, best_pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  HEART DISEASE PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # 1. Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 2. Cross-validation on all classifiers
    print("\n[STEP 1] Stratified K-Fold Cross-Validation")
    classifiers = get_classifiers()
    cv_df = run_cross_validation(classifiers, X_train, y_train)

    print("\nCV Results (sorted by ROC-AUC):")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(cv_df[["Model", "CV_ROC_AUC_Mean", "CV_ROC_AUC_Std",
                  "CV_Accuracy_Mean", "CV_F1_Mean", "CV_Recall_Mean"]].to_string(index=False))

    # 3. Hyperparameter tuning
    print("\n[STEP 2] Hyperparameter Tuning (GridSearchCV)")
    tuned_pipelines = tune_best_models(classifiers, X_train, y_train)

    # 4. Save best model
    print("\n[STEP 3] Saving Best Model")
    best_name, best_pipeline = save_best_model(tuned_pipelines)
    
    # 5. Save CV results (full evaluation is done in evaluate_models.py)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    print(f"[INFO] CV results saved to: {REPORTS_DIR / 'cv_results.csv'}")

    # Return objects for use in notebook / evaluate_models.py
    return X_train, X_test, y_train, y_test, classifiers, tuned_pipelines, cv_df, best_name


if __name__ == "__main__":
    main()

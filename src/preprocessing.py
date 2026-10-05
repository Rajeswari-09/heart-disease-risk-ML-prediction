"""
preprocessing.py
----------------
Defines the sklearn preprocessing pipeline using ColumnTransformer.

Design decisions:
  - Numerical features: median imputation → StandardScaler
    * Median is robust to outliers (e.g., extreme cholesterol values)
  - Categorical features: most-frequent imputation → OneHotEncoder
    * 'handle_unknown="ignore"' ensures unseen categories in deployment
      are encoded as all-zeros rather than raising an error
  - Pipeline connects preprocessing and classifier in a single object
    to prevent any data leakage: transformers are fit only on training data.

Data leakage prevention:
  - fit_transform() is called ONLY on the training set
  - transform() (no fit) is called on the validation/test set
  - The entire fitted pipeline is saved to disk so the Streamlit app
    applies the exact same transformations the model was trained on.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Feature definitions (mirror those in data_inspection.get_feature_lists)
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# 'ca' is stored as float but represents number of major vessels (0–3); it has
# substantial missingness (66%) and we treat it as numerical with median impute.


def build_numerical_transformer() -> Pipeline:
    """
    Numerical pipeline:
      1. Median imputation — preserves rows with missing values; median is
         resistant to extreme outliers common in clinical lab data.
      2. StandardScaler — zero mean, unit variance. Required for distance-
         based models (KNN, SVM) and beneficial for Logistic Regression.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_categorical_transformer() -> Pipeline:
    """
    Categorical pipeline:
      1. Most-frequent (mode) imputation — safe default for nominal features.
      2. OneHotEncoder with handle_unknown='ignore' — new categories seen at
         inference time are treated as all-zeros, avoiding runtime errors.
         drop='first' is NOT used here so feature importance remains
         interpretable per individual category.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor(
    numerical_features: list = None,
    categorical_features: list = None,
) -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that handles all features.

    Parameters
    ----------
    numerical_features : list, optional
        Column names for numerical features. Defaults to NUMERICAL_FEATURES.
    categorical_features : list, optional
        Column names for categorical features. Defaults to CATEGORICAL_FEATURES.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready to be inserted into a Pipeline.
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", build_numerical_transformer(), numerical_features),
            ("cat", build_categorical_transformer(), categorical_features),
        ],
        remainder="drop",       # drop id, dataset, num — non-predictive cols
        verbose_feature_names_out=True,
    )
    return preprocessor


def build_full_pipeline(classifier, numerical_features=None, categorical_features=None) -> Pipeline:
    """
    Combine preprocessor + classifier into a single sklearn Pipeline.

    This is the canonical way to prevent data leakage:
      - pipeline.fit(X_train, y_train)  → fits both preprocessor AND model
      - pipeline.predict(X_test)        → transforms X_test then predicts
    The test set is NEVER used to fit any transformer.

    Parameters
    ----------
    classifier : sklearn estimator
        The classification model to use.
    numerical_features : list, optional
    categorical_features : list, optional

    Returns
    -------
    Pipeline
        Full pipeline (preprocessor + classifier).
    """
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def get_feature_names_out(fitted_pipeline: Pipeline) -> list:
    """
    Extract transformed feature names from a fitted pipeline's preprocessor.
    Useful for feature-importance visualisation.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    from data_inspection import load_data
    df = load_data()
    preprocessor = build_preprocessor()
    print("[INFO] Preprocessor built successfully.")
    print(preprocessor)

"""
data_inspection.py
------------------
Loads the UCI Heart Disease dataset, inspects structure, prints a detailed
summary, and returns the raw DataFrame for downstream use.

Dataset location: data/heart_disease_uci.csv (relative to project root).
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path resolution – always resolve relative to this file's grandparent dir
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_uci.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return the DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded from: {path}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a comprehensive summary of the DataFrame."""

    sep = "=" * 60

    print(f"\n{sep}")
    print("  UCI HEART DISEASE DATASET — INSPECTION SUMMARY")
    print(sep)

    # ---- Shape -------------------------------------------------------
    print(f"\n{'--- Shape ---':^60}")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")

    # ---- Column names ------------------------------------------------
    print(f"\n{'--- Columns ---':^60}")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")

    # ---- Data types --------------------------------------------------
    print(f"\n{'--- Data Types ---':^60}")
    print(df.dtypes.to_string())

    # ---- Missing values ----------------------------------------------
    print(f"\n{'--- Missing Values ---':^60}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    mv = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    mv = mv[mv["Missing Count"] > 0]
    if mv.empty:
        print("  No missing values found.")
    else:
        print(mv.to_string())

    # ---- Duplicates --------------------------------------------------
    print(f"\n{'--- Duplicates ---':^60}")
    dup_count = df.duplicated().sum()
    print(f"  Duplicate rows: {dup_count}")

    # ---- Numerical columns -------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n{'--- Numerical Columns ---':^60}")
    print(f"  {num_cols}")
    print(df[num_cols].describe().round(2).to_string())

    # ---- Categorical columns -----------------------------------------
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    print(f"\n{'--- Categorical Columns ---':^60}")
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        print(f"\n  {col}:")
        print(vc.to_string(header=False).replace("^", " "))

    # ---- Target column -----------------------------------------------
    print(f"\n{'--- Target Column: num ---':^60}")
    print("  Original 'num' distribution (0=no disease, 1-4=disease severity):")
    print(df["num"].value_counts().sort_index().to_string())
    binary_target = (df["num"] > 0).astype(int)
    print("\n  Binary target (0=No Disease, 1=Disease present):")
    print(binary_target.value_counts().sort_index().to_string())
    pct_pos = binary_target.mean() * 100
    print(f"\n  Positive class (disease): {pct_pos:.1f}%")
    print(f"  Negative class (no disease): {100 - pct_pos:.1f}%")

    # ---- Dataset source column ---------------------------------------
    print(f"\n{'--- Dataset Source (dataset column) ---':^60}")
    print(df["dataset"].value_counts().to_string())

    print(f"\n{sep}")
    print("  ASSUMPTIONS & DESIGN DECISIONS")
    print(sep)
    print(
        "  1. 'id' column -- sequential patient ID, not a predictive feature. Dropped.\n"
        "  2. 'dataset' column -- study centre identifier (Cleveland, Hungary, etc.).\n"
        "     Dropped to avoid data-source leakage into the model.\n"
        "  3. 'num' (target) -- ordinal severity (0-4). Converted to binary:\n"
        "       0  -> No Heart Disease\n"
        "       1+ -> Heart Disease Present\n"
        "     This follows standard UCI benchmark practice.\n"
        "  4. 'ca' and 'thal' have very high missingness (66% and 53%). They will\n"
        "     be retained and imputed rather than dropped, because they carry\n"
        "     clinical signal, but this will be noted as a limitation.\n"
        "  5. 'fbs' and 'exang' are boolean-typed object columns -> treated as\n"
        "     categorical for One-Hot-Encoding after bool-to-string cast.\n"
        "  6. No duplicate rows detected; no action needed.\n"
        "  7. All preprocessing will be done inside an sklearn Pipeline to prevent\n"
        "     data leakage (fitting only on training data).\n"
    )


def get_feature_lists(df: pd.DataFrame):
    """
    Return lists of numerical and categorical features after dropping
    non-predictive columns and the target.
    """
    drop_cols = ["id", "dataset", "num"]
    features = [c for c in df.columns if c not in drop_cols]

    numerical = [
        c for c in features
        if df[c].dtype in [np.int64, np.float64]
    ]
    categorical = [c for c in features if c not in numerical]

    return numerical, categorical


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    num_feats, cat_feats = get_feature_lists(df)
    print(f"\nNumerical features  : {num_feats}")
    print(f"Categorical features: {cat_feats}")

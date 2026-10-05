"""
eda.py  (called from notebook; not part of the required src/ files but supports EDA section)
Generates all EDA figures and saves them to reports/figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_inspection import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
LABEL_MAP = {0: "No Disease", 1: "Disease"}


def prepare_eda_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target and return a working copy."""
    df = df.copy()
    df["target"] = (df["num"] > 0).astype(int)
    df["target_label"] = df["target"].map(LABEL_MAP)
    return df


# -------------------------------------------------------------------------
# 1. Target class distribution
# -------------------------------------------------------------------------
def plot_target_distribution(df, save=True):
    counts = df["target_label"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Bar chart
    sns.countplot(data=df, x="target_label", palette=["#2ecc71", "#e74c3c"],
                  order=["No Disease", "Disease"], ax=axes[0])
    axes[0].set_title("Target Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Heart Disease Status")
    axes[0].set_ylabel("Count")
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5,
                     f"{int(bar.get_height())} ({bar.get_height()/len(df)*100:.1f}%)",
                     ha="center", fontsize=11)

    # Pie chart
    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"], startangle=140,
                textprops={"fontsize": 11})
    axes[1].set_title("Class Split", fontsize=13, fontweight="bold")

    plt.suptitle("Heart Disease Class Distribution (Binary)", fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 01_target_distribution.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# 2. Age distribution
# -------------------------------------------------------------------------
def plot_age_distribution(df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(df["age"], bins=20, kde=True, color="steelblue", ax=axes[0])
    axes[0].set_title("Age Distribution (All Patients)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(df["age"].median(), color="red", linestyle="--",
                    label=f"Median = {df['age'].median()}")
    axes[0].legend()

    sns.boxplot(data=df, x="target_label", y="age",
                palette=["#2ecc71", "#e74c3c"],
                order=["No Disease", "Disease"], ax=axes[1])
    axes[1].set_title("Age vs Heart Disease Status", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Heart Disease Status")
    axes[1].set_ylabel("Age (years)")

    plt.suptitle("Age Distribution Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "02_age_distribution.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 02_age_distribution.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# 3. Cholesterol distribution
# -------------------------------------------------------------------------
def plot_cholesterol_distribution(df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    chol_valid = df["chol"].dropna()
    sns.histplot(chol_valid, bins=30, kde=True, color="darkorange", ax=axes[0])
    axes[0].set_title("Serum Cholesterol Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Cholesterol (mg/dl)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(chol_valid.median(), color="red", linestyle="--",
                    label=f"Median = {chol_valid.median():.0f}")
    axes[0].legend()

    sns.boxplot(data=df, x="target_label", y="chol",
                palette=["#2ecc71", "#e74c3c"],
                order=["No Disease", "Disease"], ax=axes[1])
    axes[1].set_title("Cholesterol vs Heart Disease", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Heart Disease Status")
    axes[1].set_ylabel("Cholesterol (mg/dl)")

    plt.suptitle("Cholesterol Distribution Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "03_cholesterol_distribution.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 03_cholesterol_distribution.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# 4. Resting blood pressure
# -------------------------------------------------------------------------
def plot_bp_distribution(df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bp_valid = df["trestbps"].dropna()
    sns.histplot(bp_valid, bins=25, kde=True, color="crimson", ax=axes[0])
    axes[0].set_title("Resting Blood Pressure Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Resting BP (mm Hg)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(bp_valid.median(), color="navy", linestyle="--",
                    label=f"Median = {bp_valid.median():.0f}")
    axes[0].legend()

    sns.violinplot(data=df, x="target_label", y="trestbps",
                   palette=["#2ecc71", "#e74c3c"],
                   order=["No Disease", "Disease"], ax=axes[1])
    axes[1].set_title("Resting BP vs Heart Disease", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Heart Disease Status")
    axes[1].set_ylabel("Resting BP (mm Hg)")

    plt.suptitle("Resting Blood Pressure Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "04_bp_distribution.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 04_bp_distribution.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# 5. All numerical feature distributions
# -------------------------------------------------------------------------
def plot_numerical_distributions(df, save=True):
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        valid = df[col].dropna()
        sns.histplot(valid, bins=25, kde=True, ax=axes[i], color=sns.color_palette("tab10")[i])
        axes[i].set_title(f"{col} Distribution", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].axvline(valid.mean(), color="red", linestyle="--", linewidth=1.2,
                        label=f"Mean={valid.mean():.1f}")
        axes[i].legend(fontsize=8)

    plt.suptitle("Numerical Feature Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "05_numerical_distributions.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 05_numerical_distributions.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# 6. Correlation matrix
# -------------------------------------------------------------------------
def plot_correlation_matrix(df, save=True):
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "target"]
    corr_df = df[num_cols].dropna()
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix (Numerical Features)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "06_correlation_matrix.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 06_correlation_matrix.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# 7. Categorical features vs target
# -------------------------------------------------------------------------
def plot_categorical_vs_target(df, save=True):
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df["target_label"], normalize="index") * 100
        ct.plot(kind="bar", ax=axes[i], color=["#2ecc71", "#e74c3c"],
                edgecolor="black", width=0.6)
        axes[i].set_title(f"{col} vs Heart Disease (%)", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Percentage (%)")
        axes[i].set_ylim(0, 110)
        axes[i].legend(title="Status", fontsize=8)
        axes[i].tick_params(axis="x", rotation=20)

    # Hide unused subplots
    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Features vs Heart Disease Status", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "07_categorical_vs_target.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 07_categorical_vs_target.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# 8. Numerical features vs target (box plots)
# -------------------------------------------------------------------------
def plot_numerical_vs_target(df, save=True):
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x="target_label", y=col,
                    palette=["#2ecc71", "#e74c3c"],
                    order=["No Disease", "Disease"],
                    ax=axes[i])
        axes[i].set_title(f"{col} vs Heart Disease", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Heart Disease Status")
        axes[i].set_ylabel(col)

    plt.suptitle("Numerical Features vs Heart Disease Status", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "08_numerical_vs_target.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 08_numerical_vs_target.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# 9. Key relationships: Chest pain type vs Thalach coloured by target
# -------------------------------------------------------------------------
def plot_key_relationships(df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: thalch vs oldpeak coloured by target
    for label, grp in df.groupby("target_label"):
        color = "#e74c3c" if label == "Disease" else "#2ecc71"
        axes[0].scatter(grp["thalch"], grp["oldpeak"], label=label,
                        alpha=0.5, color=color, edgecolors="none", s=25)
    axes[0].set_xlabel("Max Heart Rate Achieved (thalch)", fontsize=11)
    axes[0].set_ylabel("ST Depression (oldpeak)", fontsize=11)
    axes[0].set_title("Max Heart Rate vs ST Depression\nColoured by Disease Status",
                      fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Violin: age by chest pain type and target
    sns.violinplot(data=df, x="cp", y="age", hue="target_label",
                   palette={"No Disease": "#2ecc71", "Disease": "#e74c3c"},
                   split=True, ax=axes[1], inner="quartile")
    axes[1].set_title("Age Distribution by Chest Pain Type\nand Disease Status",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Chest Pain Type")
    axes[1].set_ylabel("Age (years)")
    axes[1].tick_params(axis="x", rotation=15)

    plt.suptitle("Key Feature Relationships", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "09_key_relationships.png", dpi=150, bbox_inches="tight")
        print("[SAVED] 09_key_relationships.png")
    plt.close(fig)


# -------------------------------------------------------------------------
# Master function
# -------------------------------------------------------------------------
def run_eda(df: pd.DataFrame = None) -> None:
    """Run all EDA plots. Loads data if not provided."""
    if df is None:
        df = load_data()
    df = prepare_eda_df(df)

    print("\n[EDA] Generating figures -> reports/figures/")
    plot_target_distribution(df)
    plot_age_distribution(df)
    plot_cholesterol_distribution(df)
    plot_bp_distribution(df)
    plot_numerical_distributions(df)
    plot_correlation_matrix(df)
    plot_categorical_vs_target(df)
    plot_numerical_vs_target(df)
    plot_key_relationships(df)
    print("[EDA] All figures saved.\n")


if __name__ == "__main__":
    run_eda()

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _save_fig(fig, save_dir: Path | None, filename: str) -> None:
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / filename, bbox_inches="tight", dpi=150)


def run_eda(df: pd.DataFrame, save_dir: Path | None = None) -> pd.DataFrame:
    """Generate basic EDA plots and save them."""

    # Target distribution
    if "Churn" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df["Churn"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Churn Distribution")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Count")
        _save_fig(fig, save_dir, "eda_churn_distribution.png")
        plt.show()
        plt.close(fig)

    # Monthly charges
    if "MonthlyCharges" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df["MonthlyCharges"].astype(float).plot(kind="hist", bins=30, ax=ax)
        ax.set_title("Monthly Charges Histogram")
        ax.set_xlabel("MonthlyCharges")
        _save_fig(fig, save_dir, "eda_monthlycharges_hist.png")
        plt.show()
        plt.close(fig)

    # Tenure
    if "tenure" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df["tenure"].astype(float).plot(kind="hist", bins=30, ax=ax)
        ax.set_title("Tenure Histogram")
        ax.set_xlabel("Tenure")
        _save_fig(fig, save_dir, "eda_tenure_hist.png")
        plt.show()
        plt.close(fig)

    return df


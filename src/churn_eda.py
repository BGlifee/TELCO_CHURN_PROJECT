# churn_eda.py
"""
Churn EDA + driver analysis utilities (run BEFORE modeling)
- Target overview
- Categorical drivers: churn rate table, chi-square test, Cramér’s V effect size
- Numeric drivers: summary stats by target, simple univariate AUC, optional tests
- Saves tables (.csv) and plots (.png) to save_dir

Usage:
    from churn_eda import run_churn_eda
    report = run_churn_eda(X_train, y_train, save_dir="results/eda")

Returns:
    report dict with key DataFrames
"""

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

# Optional stats (graceful fallback)
try:
    from scipy.stats import chi2_contingency, mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

def _ensure_dir(d: str | Path) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _safe_binary(y: pd.Series) -> pd.Series:
    """Convert common churn labels to 0/1 safely."""
    if y.dtype == bool:
        return y.astype(int)
    if y.dtype == object:
        # common Telco labels: "Yes"/"No"
        m = y.astype(str).str.strip().str.lower()
        if set(m.unique()) <= {"yes", "no"}:
            return (m == "yes").astype(int)
    # assume already 0/1
    return y.astype(int)

def _cramers_v(confusion: pd.DataFrame) -> float:
    """
    Cramér’s V effect size for association between categorical feature and target.
    Range: 0 (none) ~ 1 (strong)
    """
    if confusion.size == 0:
        return np.nan
    obs = confusion.to_numpy()
    n = obs.sum()
    if n == 0:
        return np.nan

    # If scipy available, use chi2; else approximate from expected via numpy (still ok)
    if SCIPY_AVAILABLE:
        chi2, _, _, _ = chi2_contingency(obs, correction=False)
    else:
        # manual chi-square without correction
        row_sums = obs.sum(axis=1, keepdims=True)
        col_sums = obs.sum(axis=0, keepdims=True)
        expected = row_sums @ col_sums / n
        expected = np.where(expected == 0, 1e-12, expected)
        chi2 = ((obs - expected) ** 2 / expected).sum()

    r, k = obs.shape
    denom = n * (min(r - 1, k - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))

def target_overview(y: pd.Series) -> pd.DataFrame:
    yb = _safe_binary(y)
    out = pd.DataFrame({
        "count": [len(yb)],
        "churn_1_count": [int(yb.sum())],
        "churn_rate": [float(yb.mean())],
        "non_churn_0_count": [int((1 - yb).sum())],
    })
    return out

def categorical_driver_table(
    X: pd.DataFrame,
    y: pd.Series,
    col: str,
    min_count: int = 30
) -> pd.DataFrame:
    """
    For one categorical feature:
    - count
    - churn_rate
    - lift_vs_overall
    - chi2 p-value (if scipy)
    - Cramér’s V
    """
    yb = _safe_binary(y)
    s = X[col].astype("object").fillna("MISSING")

    overall = float(yb.mean())
    grp = pd.DataFrame({"x": s, "y": yb}).groupby("x")["y"].agg(["count", "mean"]).rename(columns={"mean": "churn_rate"})
    grp = grp.sort_values("churn_rate", ascending=False)

    # filter rare levels (but keep them in a combined bucket for interpretability)
    rare_mask = grp["count"] < min_count
    if rare_mask.any():
        rare_levels = set(grp.index[rare_mask])
        s2 = s.where(~s.isin(rare_levels), other="__RARE__")
        grp = pd.DataFrame({"x": s2, "y": yb}).groupby("x")["y"].agg(["count", "mean"]).rename(columns={"mean": "churn_rate"})
        grp = grp.sort_values("churn_rate", ascending=False)

    grp["lift_vs_overall"] = grp["churn_rate"] - overall
    grp["lift_ratio"] = grp["churn_rate"] / overall if overall > 0 else np.nan

    # Association stats
    conf = pd.crosstab(s.fillna("MISSING"), yb)
    v = _cramers_v(conf)
    p = np.nan
    if SCIPY_AVAILABLE and conf.shape[0] > 1 and conf.shape[1] > 1:
        _, p, _, _ = chi2_contingency(conf, correction=False)

    grp["chi2_p_value"] = p
    grp["cramers_v"] = v
    grp.insert(0, "feature", col)
    grp = grp.reset_index().rename(columns={"x": "level"})
    return grp

def numeric_driver_table(
    X: pd.DataFrame,
    y: pd.Series,
    col: str
) -> pd.DataFrame:
    """
    For one numeric feature:
    - summary stats by class
    - univariate AUC using the raw feature as score
    - optional Mann–Whitney U p-value (if scipy)
    """
    yb = _safe_binary(y)
    x = pd.to_numeric(X[col], errors="coerce")
    df = pd.DataFrame({"x": x, "y": yb}).dropna()

    out = {"feature": col, "n": int(len(df))}
    if len(df) == 0 or df["y"].nunique() < 2:
        return pd.DataFrame([out])

    # Stats by class
    for cls in [0, 1]:
        xs = df.loc[df["y"] == cls, "x"]
        out[f"mean_y{cls}"] = float(xs.mean())
        out[f"median_y{cls}"] = float(xs.median())
        out[f"std_y{cls}"] = float(xs.std(ddof=1))
        out[f"q25_y{cls}"] = float(xs.quantile(0.25))
        out[f"q75_y{cls}"] = float(xs.quantile(0.75))

    # Univariate AUC (direction can flip)
    auc = roc_auc_score(df["y"], df["x"])
    out["univariate_auc"] = float(max(auc, 1 - auc))  # make it >=0.5 for easy ranking
    out["auc_direction_flipped"] = bool(auc < 0.5)

    # Mann–Whitney U (non-parametric)
    p = np.nan
    if SCIPY_AVAILABLE:
        x0 = df.loc[df["y"] == 0, "x"]
        x1 = df.loc[df["y"] == 1, "x"]
        if len(x0) > 0 and len(x1) > 0:
            _, p = mannwhitneyu(x0, x1, alternative="two-sided")
    out["mw_p_value"] = p

    return pd.DataFrame([out])

def _plot_categorical_top_levels(df_cat: pd.DataFrame, save_path: Path, top_k: int = 12):
    """
    df_cat must include: level, churn_rate, count
    """
    d = df_cat.copy()
    d["label"] = d["level"].astype(str) + " (n=" + d["count"].astype(int).astype(str) + ")"
    d = d.sort_values("churn_rate", ascending=False).head(top_k)

    plt.figure(figsize=(9, 4))
    plt.bar(d["label"], d["churn_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Churn rate")
    plt.title(f"Top levels by churn rate: {d['feature'].iloc[0]}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _plot_numeric_box(X: pd.DataFrame, y: pd.Series, col: str, save_path: Path):
    yb = _safe_binary(y)
    x = pd.to_numeric(X[col], errors="coerce")
    df = pd.DataFrame({"x": x, "y": yb}).dropna()
    if df.empty:
        return

    data0 = df.loc[df["y"] == 0, "x"].to_numpy()
    data1 = df.loc[df["y"] == 1, "x"].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.boxplot([data0, data1], labels=["No churn (0)", "Churn (1)"])
    plt.ylabel(col)
    plt.title(f"{col} distribution by churn")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_churn_eda(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: str | Path = "results/eda",
    *,
    known_categorical: list[str] | None = None,
    known_numeric: list[str] | None = None,
    min_count: int = 30
) -> dict[str, pd.DataFrame]:
    """
    Main entry: runs EDA/driver analysis and saves outputs.
    Returns report dict of DataFrames.
    """
    save_dir = _ensure_dir(save_dir)
    yb = _safe_binary(y)

    # Detect column types (override-able)
    if known_numeric is None:
        # Treat "number-like" as numeric (but exclude id-like)
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        numeric_cols = [c for c in known_numeric if c in X.columns]

    if known_categorical is None:
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
    else:
        categorical_cols = [c for c in known_categorical if c in X.columns]

    # 1) Target overview
    overview = target_overview(yb)
    overview.to_csv(save_dir / "target_overview.csv", index=False)

    # 2) Categorical drivers
    cat_tables = []
    cat_feature_summary = []
    for col in categorical_cols:
        try:
            t = categorical_driver_table(X, yb, col, min_count=min_count)
            cat_tables.append(t)

            # feature-level summary: strongest lift level + cramers v
            top = t.sort_values("churn_rate", ascending=False).head(1)
            cat_feature_summary.append({
                "feature": col,
                "cramers_v": float(t["cramers_v"].dropna().iloc[0]) if t["cramers_v"].notna().any() else np.nan,
                "chi2_p_value": float(t["chi2_p_value"].dropna().iloc[0]) if t["chi2_p_value"].notna().any() else np.nan,
                "top_level": str(top["level"].iloc[0]),
                "top_level_churn_rate": float(top["churn_rate"].iloc[0]),
                "top_level_count": int(top["count"].iloc[0]),
            })

            # plot per categorical feature (top levels)
            _plot_categorical_top_levels(
                t[["feature","level","count","churn_rate"]].copy(),
                save_dir / f"cat_{col}_top_levels.png",
                top_k=12
            )
        except Exception:
            continue

    cat_levels = pd.concat(cat_tables, ignore_index=True) if cat_tables else pd.DataFrame()
    cat_summary = pd.DataFrame(cat_feature_summary).sort_values("cramers_v", ascending=False) if cat_feature_summary else pd.DataFrame()

    cat_levels.to_csv(save_dir / "categorical_levels.csv", index=False)
    cat_summary.to_csv(save_dir / "categorical_feature_summary.csv", index=False)

    # 3) Numeric drivers
    num_tables = []
    for col in numeric_cols:
        try:
            t = numeric_driver_table(X, yb, col)
            num_tables.append(t)
            _plot_numeric_box(X, yb, col, save_dir / f"num_{col}_boxplot.png")
        except Exception:
            continue

    num_summary = pd.concat(num_tables, ignore_index=True) if num_tables else pd.DataFrame()
    if not num_summary.empty and "univariate_auc" in num_summary.columns:
        num_summary = num_summary.sort_values("univariate_auc", ascending=False)
    num_summary.to_csv(save_dir / "numeric_feature_summary.csv", index=False)

    report = {
        "target_overview": overview,
        "categorical_levels": cat_levels,
        "categorical_feature_summary": cat_summary,
        "numeric_feature_summary": num_summary,
    }
    return report

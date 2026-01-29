from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class BusinessParams:
    threshold: float
    high_risk_p: float = 0.70
    remaining_months: int = 12
    retention_uplift: float = 0.30
    top_n: int = 100

def build_full_features(df: pd.DataFrame, total_median: float) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(total_median)

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["service_count"] = (df[service_cols] == "Yes").sum(axis=1)

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"]
    )

    X_all = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])
    return X_all


def score_customers_full(model, df_full: pd.DataFrame, params: BusinessParams, total_median: float):
    X_all = build_full_features(df_full, total_median=total_median)

    probs_all = model.predict_proba(X_all)[:, 1]
    preds_all = (probs_all >= params.threshold).astype(int)

    scored = df_full.copy()
    scored["churn_proba"] = probs_all
    scored["churn_pred"] = preds_all

    if "Churn" in scored.columns:
        scored["Churn_actual"] = (scored["Churn"] == "Yes").astype(int)

    scored["risk_segment"] = pd.cut(
        scored["churn_proba"],
        [-0.01, 0.3, 0.7, 1.0],
        labels=["Low", "Medium", "High"]
    )

    scored["expected_revenue_loss"] = (
        scored["churn_proba"] * scored["MonthlyCharges"] * params.remaining_months
    )
    scored["retention_opportunity"] = scored["expected_revenue_loss"] * params.retention_uplift

    kpi = pd.DataFrame([{
        "TotalCustomers": len(scored),
        "PredictedChurnRate": float(scored["churn_pred"].mean()),
        "HighRiskCustomerRate": float((scored["churn_proba"] >= params.high_risk_p).mean()),
        "ExpectedRevenueLoss($)": float(scored["expected_revenue_loss"].sum()),
        "RetentionOpportunity($)": float(scored["retention_opportunity"].sum()),
        "ThresholdUsed": float(params.threshold),
    }])

    risk_summary = scored.groupby("risk_segment", observed=True).agg(
        customers=("customerID", "count"),
        expected_loss=("expected_revenue_loss", "sum"),
    ).reset_index()

    high = scored[scored["churn_proba"] >= params.high_risk_p]
    topN = scored.sort_values("expected_revenue_loss", ascending=False).head(params.top_n)

    total_loss = scored["expected_revenue_loss"].sum()
    high_loss = high["expected_revenue_loss"].sum()

    insight = pd.DataFrame([{
        "HighRiskCustomerPct": float(len(high) / len(scored)) if len(scored) else 0.0,
        "HighRiskLossShare": float(high_loss / total_loss) if total_loss else 0.0,
        "HighRiskAvgLoss": float(high["expected_revenue_loss"].mean()) if len(high) else 0.0,
        f"Top{params.top_n}LossShare": float(topN["expected_revenue_loss"].sum() / total_loss) if total_loss else 0.0,
    }])

    return scored, kpi, risk_summary, insight

def export_powerbi_assets(
    results_dir: str | Path,
    scored: pd.DataFrame,
    kpi: pd.DataFrame,
    risk_summary: pd.DataFrame,
    insight: pd.DataFrame,
    params: BusinessParams,
    prefix: str = "full",
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    scored.to_csv(results_dir / f"scored_customers_{prefix}.csv", index=False)
    kpi.to_csv(results_dir / f"kpi_summary_{prefix}.csv", index=False)
    risk_summary.to_csv(results_dir / f"risk_summary_{prefix}.csv", index=False)
    insight.to_csv(results_dir / f"insight_summary_{prefix}.csv", index=False)

    pd.Series({
        "threshold": float(params.threshold),
        "high_risk_p": float(params.high_risk_p),
        "remaining_months": int(params.remaining_months),
        "retention_uplift": float(params.retention_uplift),
        "top_n": int(params.top_n),
    }).to_json(results_dir / "business_assumptions.json")

    return results_dir

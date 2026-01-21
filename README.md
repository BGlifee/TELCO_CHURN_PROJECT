# TELCO_CHURN_PROJECT
<p align="center">
  <img src="assets/dashboard_demo.jpg" width="900">
</p>

**Customer Churn Prediction & Business Risk Scoring**

Retention budgets in telecom are finite, and mis-targeted interventions directly translate into lost revenue.
This project develops a churn prediction system that not only identifies high-risk customers, but also converts model predictions into concrete financial trade-offs.
By evaluating Accuracy, F1-score, and AUC alongside False Negative and False Positive cost analysis, the project determines an optimal churn threshold that maximizes expected profit under real-world budget constraints. The pipeline deliver clear, quantitative evidence to guide retention investment and decision-making.

---

## What It Does

* Predicts customer churn using supervised learning
* Generates churn **probabilities** (not just binary labels)
* Segments customers into actionable risk tiers
* Estimates expected revenue loss and retention opportunity
* Automatically produces evaluation metrics and plots

---

## Workflow

Data ingestion → EDA → Preprocessing → Model training →
Probability scoring → Risk segmentation → Business impact estimation

---

## Model Evaluation

Model performance is evaluated using **ROC-AUC** to measure overall discriminatory power between churn and non-churn customers.

Because churn prediction is a **decision problem**, predicted probabilities are converted into labels using a configurable **decision threshold (default = 0.50)**.
Threshold-dependent metrics such as **Precision, Recall, and F1-score** are used to evaluate performance under different business trade-offs.

Threshold tuning allows the business to balance:

* False positives (unnecessary retention cost)
* False negatives (missed churners)

---

## Risk Segmentation

Customers are segmented based on predicted churn probability:

* **Low Risk:** < 0.30
* **Medium Risk:** 0.30 – 0.70
* **High Risk:** ≥ 0.70

This probability-based approach enables prioritization of retention efforts instead of treating all churn predictions equally.

---

## Business Impact & ROI Estimation

Expected revenue loss is estimated as:

```
Expected Revenue Loss
= P(churn) × Monthly Charges × Remaining Months
```

A conservative **retention uplift rate (30%)** is applied to estimate potential recovery value:

```
Retention Opportunity
= Expected Revenue Loss × Retention Uplift
```

This links model outputs directly to **financial impact and ROI**, rather than relying solely on abstract accuracy metrics.

---

## Key Business Insights

* A small **high-risk customer segment** accounts for a disproportionate share of expected revenue loss
* Probability-based scoring enables **targeted retention strategies** with higher expected return
* Threshold selection provides a direct control lever between operational cost and churn prevention effectiveness

---

## Outputs

* `scored_customers.csv` — customer-level churn probabilities and risk segments
* `kpi_summary.csv` — churn rate, high-risk rate, revenue KPIs
* `risk_summary.csv` — risk distribution and revenue exposure
* `insight_summary.csv` — concentration of churn risk and loss
* ROC, Confusion Matrix, Precision–Recall plots (saved to `results/graphs/`)

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Jupyter

---

## Notes

This project emphasizes **interpretability, reproducibility, and business relevance** over black-box optimization and is designed to be easily extended to dashboards or production scoring pipelines.

---






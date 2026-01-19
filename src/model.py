from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)


def _save_fig(fig, save_dir: Path | None, filename: str) -> None:
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / filename, bbox_inches="tight", dpi=150)


def train_and_evaluate(X_train, X_test, y_train, y_test, save_dir: Path | None = None):
    """
    Train model, evaluate, and save evaluation plots.
    """

    # Train model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_t = thresholds[best_idx]

    preds = (probs >= best_t).astype(int)

    # ROC curve
    auc = roc_auc_score(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    _save_fig(fig, save_dir, "model_roc_curve.png")
    plt.show()
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix (threshold = {best_t:.3f})")
    _save_fig(fig, save_dir, "model_confusion_matrix.png")
    plt.show()
    plt.close(fig)

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    _save_fig(fig, save_dir, "model_precision_recall_curve.png")
    plt.show()
    plt.close(fig)

    return model, preds, probs, best_t
    
             
"""
Model training and evaluation utilities.
- Builds a scikit-learn pipeline (ColumnTransformer + estimator)
- Uses SMOTE (when imblearn is available) inside an imbalanced-learn Pipeline
- Runs RandomizedSearchCV optimizing F1 for both RandomForest and LightGBM (separate searches)
- Selects best estimator between the two
- Calibrates the chosen estimator with CalibratedClassifierCV (isotonic)
- Produces out-of-fold probabilities to tune decision threshold (maximize F1)
- Saves basic evaluation plots (ROC, Precision-Recall)

Function: train_and_evaluate(X_train, X_test, y_train, y_test, save_dir)
Returns: trained_model, test_preds, test_probs, best_threshold
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# imbalanced-learn is optional; we prefer SMOTE in-pipeline when available
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    from sklearn.pipeline import Pipeline as SKPipeline
    IMBLEARN_AVAILABLE = False

def _ensure_dir(d: Path):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _random_search_for_estimator(pipe, param_distributions, X_train, y_train, cv, n_iter, random_state):
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )
    rs.fit(X_train, y_train)
    return rs

def train_and_evaluate(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       save_dir: str | Path = 'results/graphs',
                       random_state: int = 42,
                       n_iter: int = 40):
    """Train models (RF and LightGBM), calibrate best estimator, tune threshold, and evaluate.

    Returns:
        best_trained_model, test_predictions, test_probabilities, best_threshold
    """
    save_dir = _ensure_dir(Path(save_dir))

    # Identify numeric and categorical columns
    numeric_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'service_count'] if c in X_train.columns]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # Preprocessing
    num_pipe = StandardScaler()
    try :
        cat_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        cat_pipe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preproc = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ---- Random Forest search ----
    rf_clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    if IMBLEARN_AVAILABLE:
        rf_pipe = ImbPipeline([('preproc', preproc), ('smote', SMOTE(random_state=random_state)), ('clf', rf_clf)])
    else:
        rf_pipe = SKPipeline([('preproc', preproc), ('clf', rf_clf)])

    rf_params = {
        'clf__n_estimators': [200, 400, 800],
        'clf__max_depth': [6, 12, 20, None],
        'clf__min_samples_split': [2, 5, 10]
    }
    if IMBLEARN_AVAILABLE:
        rf_params['smote__sampling_strategy'] = [0.5, 0.7, 0.9]

    print('\nRunning Random Forest randomized search...')
    rs_rf = _random_search_for_estimator(rf_pipe, rf_params, X_train, y_train, cv, n_iter // 2, random_state)

    # ---- LightGBM search (if available) ----
    rs_lgb = None
    if LGB_AVAILABLE:
        lgb_clf = lgb.LGBMClassifier(random_state=random_state, n_jobs=-1)
        if IMBLEARN_AVAILABLE:
            lgb_pipe = ImbPipeline([('preproc', preproc), ('smote', SMOTE(random_state=random_state)), ('clf', lgb_clf)])
        else:
            lgb_pipe = SKPipeline([('preproc', preproc), ('clf', lgb_clf)])

        lgb_params = {
            'clf__n_estimators': [200, 400, 800],
            'clf__learning_rate': [0.01, 0.03, 0.05, 0.1],
            'clf__num_leaves': [31, 50, 100],
            'clf__max_depth': [-1, 6, 12, 20]
        }
        if IMBLEARN_AVAILABLE:
            lgb_params['smote__sampling_strategy'] = [0.5, 0.7, 0.9]

        print('\nRunning LightGBM randomized search...')
        rs_lgb = _random_search_for_estimator(lgb_pipe, lgb_params, X_train, y_train, cv, n_iter // 2, random_state)
    else:
        print('\nLightGBM not available in environment; skipping LGB search.')

    # Choose best between RF and LGB (compare best CV F1)
    best_estimator = rs_rf.best_estimator_
    best_cv_score = rs_rf.best_score_
    chosen = 'random_forest'
    best_params = rs_rf.best_params_

    if rs_lgb is not None and rs_lgb.best_score_ > best_cv_score:
        best_estimator = rs_lgb.best_estimator_
        best_cv_score = rs_lgb.best_score_
        chosen = 'lightgbm'
        best_params = rs_lgb.best_params_

    print(f"\nChosen model: {chosen} with CV F1 = {best_cv_score:.4f}")
    print('Best params:', best_params)

    # Calibrate the chosen estimator for better probabilities
    try:
        calibrated = CalibratedClassifierCV(best_estimator, cv=cv, method='isotonic')
        calibrated.fit(X_train, y_train)
        final_model = calibrated
        calibrated_flag = True
    except Exception as e:
        print('Calibration failed or not supported; using uncalibrated model. Error:', e)
        # Refit best_estimator on full training data (it may already be refit by RandomizedSearchCV)
        try:
            best_estimator.fit(X_train, y_train)
        except Exception:
            pass
        final_model = best_estimator
        calibrated_flag = False

    # Out-of-fold probabilities on training set for reliable threshold tuning
    try:
        oof_probs = cross_val_predict(final_model, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    except Exception:
        # Fallback: use predict_proba on training data (less reliable)
        oof_probs = final_model.predict_proba(X_train)[:, 1]

    # Find threshold that maximizes F1 on OOF predictions
    thresholds = np.linspace(0.0, 1.0, 101)
    f1s = [f1_score(y_train, (oof_probs >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])

    # Evaluate on test set
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_t).astype(int)

    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc = roc_auc_score(y_test, test_probs)

    print('\nModel Performance (with tuned threshold)')
    print('Chosen model:', chosen)
    print('Calibrated probabilities:' , calibrated_flag)
    print('Best threshold:', best_t)
    print('Accuracy: {:.4f}'.format(acc))
    print('F1 Score: {:.4f}'.format(f1))
    print('ROC AUC Score: {:.4f}'.format(roc))
    print('\nClassification report:\n', classification_report(y_test, test_preds))

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

    # Precision-Recall curve
    precision, recall, pr_thresh = precision_recall_curve(y_test, test_probs)
    ap = average_precision_score(y_test, test_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()

    # Confusion matrix saved as text and plot
    cm = confusion_matrix(y_test, test_preds)
    cm_df = pd.DataFrame(cm, index=['No', 'Yes'], columns=['Pred_No', 'Pred_Yes'])
    cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No', 'Yes'])
    plt.yticks(tick_marks, ['No', 'Yes'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Save training metadata
    metadata = {
        'chosen_model': chosen,
        'lgb_available': LGB_AVAILABLE,
        'imblearn_available': IMBLEARN_AVAILABLE,
        'best_params': best_params,
        'best_cv_score_f1': best_cv_score,
        'best_threshold': best_t,
        'test_accuracy': acc,
        'test_f1': f1,
        'test_roc_auc': roc,
        'calibrated': calibrated_flag
    }
    pd.Series(metadata).to_json(os.path.join(save_dir, 'train_metadata.json'))

    return final_model, test_preds, test_probs, best_t

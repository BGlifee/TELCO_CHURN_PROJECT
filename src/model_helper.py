
from sklearn.metrics import f1_score
import numpy as np

def find_best_threshold(y_true, probs, metric=f1_score):
    best_t = 0.5
    best_score = -1.0

    for t in np.arange(0.01, 1.0, 0.01):  # NOTE: arange, not arrange
        preds = (probs >= t).astype(int)
        s = metric(y_true, preds)
        if s > best_score:
            best_score = s
            best_t = t

    return best_t, best_score
# Metrics for comparing the two systems.
# AI use: written with help from Claude (Anthropic), see README.
#
# Accuracy alone is misleading here because 70% of cars are 'unacc',
# so I mainly use macro F1 (every class counts the same).

import random
from math import comb

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from data import CLASSES


def classification_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0),
    }
    for i, c in enumerate(CLASSES):
        out[f"precision_{c}"] = p[i]
        out[f"recall_{c}"] = r[i]
        out[f"f1_{c}"] = f[i]
    return out


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=CLASSES)


def mcnemar_exact(y_true, pred_a, pred_b):
    # exact McNemar test. b = A right/B wrong, c = A wrong/B right
    b = sum(1 for t, a, bb in zip(y_true, pred_a, pred_b) if a == t and bb != t)
    c = sum(1 for t, a, bb in zip(y_true, pred_a, pred_b) if a != t and bb == t)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


def bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=2000, seed=7):
    # 95% bootstrap confidence interval
    rng = random.Random(seed)
    idx = list(range(len(y_true)))
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in idx]
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        if metric == "macro_f1":
            vals.append(f1_score(yt, yp, labels=CLASSES, average="macro", zero_division=0))
        else:
            vals.append(accuracy_score(yt, yp))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]

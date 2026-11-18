from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"gini": None, "ks": None}
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    return {
        "gini": float(2 * roc_auc_score(y, p) - 1),
        "ks": float(np.max(tpr - fpr)),
    }


def woe_iv_by_bin(y_true: np.ndarray, bins: np.ndarray, bad_label: int = 1) -> list[dict[str, float | int]]:
    """Independently calculate bin WoE/IV (good-to-bad odds; zero cells contribute zero)."""
    y = np.asarray(y_true, dtype=int)
    groups = np.asarray(bins)
    total_good = int(np.sum(y != bad_label))
    total_bad = int(np.sum(y == bad_label))
    result = []
    for group in np.unique(groups):
        mask = groups == group
        good = int(np.sum(y[mask] != bad_label))
        bad = int(np.sum(y[mask] == bad_label))
        good_dist = good / total_good if total_good else 0.0
        bad_dist = bad / total_bad if total_bad else 0.0
        if good_dist and bad_dist:
            woe = float(np.log(good_dist / bad_dist))
            iv = float((good_dist - bad_dist) * woe)
        else:
            woe = 0.0
            iv = 0.0
        result.append({"bin": int(group), "good": good, "bad": bad, "woe": woe, "iv": iv})
    return result


def population_stability(reference: np.ndarray, current: np.ndarray) -> float | None:
    """PSI with a 0.5-count adjustment to keep empty bins finite."""
    first, second = np.asarray(reference), np.asarray(current)
    if not len(first) or not len(second):
        return None
    categories = np.union1d(np.unique(first), np.unique(second))
    counts_a = np.array([np.sum(first == category) for category in categories], dtype=float) + 0.5
    counts_b = np.array([np.sum(second == category) for category in categories], dtype=float) + 0.5
    share_a, share_b = counts_a / counts_a.sum(), counts_b / counts_b.sum()
    return float(np.sum((share_b - share_a) * np.log(share_b / share_a)))


def numeric_population_stability(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float | None:
    """PSI for a numeric measure using quantile cut points learned from the reference."""
    first, second = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    first, second = first[np.isfinite(first)], second[np.isfinite(second)]
    if not len(first) or not len(second):
        return None
    edges = np.unique(np.quantile(first, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 1:
        return 0.0
    return population_stability(np.digitize(first, edges[1:-1]), np.digitize(second, edges[1:-1]))

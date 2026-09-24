"""
fairness_metrics.py
--------------------
Computes group fairness metrics for loan eligibility models.
Metrics: Selection Rate (PPR), TPR, FPR, FNR, Disparate Impact,
         Demographic Parity, Equalized Odds
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _group_rates(y_true, y_pred, group_mask):
    """Return TPR, FPR, FNR, PPR for a boolean group mask."""
    yt = np.array(y_true)[group_mask]
    yp = np.array(y_pred)[group_mask]
    if len(yt) == 0:
        return {"tpr": np.nan, "fpr": np.nan, "fnr": np.nan, "ppr": np.nan, "n": 0}
    tp = np.sum((yt == 1) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    fp = np.sum((yt == 0) & (yp == 1))
    tn = np.sum((yt == 0) & (yp == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # fall-out
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # miss rate = 1 - TPR
    ppr = (tp + fp) / len(yp)                          # selection / approval rate
    return {"tpr": tpr, "fpr": fpr, "fnr": fnr, "ppr": ppr, "n": len(yt)}


def compute_fairness_metrics(y_true, y_pred, sensitive_col, df):
    """
    Compute fairness metrics for each unique group in sensitive_col.

    Parameters
    ----------
    y_true : array-like, binary ground truth
    y_pred : array-like, binary predictions
    sensitive_col : str, column name in df
    df : pd.DataFrame aligned with y_true / y_pred

    Returns
    -------
    pd.DataFrame with per-group metrics + overall fairness scores
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    groups = df[sensitive_col].unique()

    rows = []
    for grp in sorted(groups):
        mask = (df[sensitive_col] == grp).values
        r = _group_rates(y_true, y_pred, mask)
        accuracy = np.mean(y_true[mask] == y_pred[mask]) if mask.sum() > 0 else np.nan
        rows.append({
            "Group":                  grp,
            "N":                      r["n"],
            "Selection Rate (PPR)":   round(r["ppr"], 4),
            "TPR (Recall)":           round(r["tpr"], 4),
            "FPR":                    round(r["fpr"], 4),
            "FNR":                    round(r["fnr"], 4),
            "Accuracy":               round(accuracy, 4),
        })

    result = pd.DataFrame(rows)

    # ── Disparate Impact (DI) ─────────────────────────────────────────────
    pprs = result["Selection Rate (PPR)"].values
    min_ppr = np.nanmin(pprs)
    max_ppr = np.nanmax(pprs)
    di = min_ppr / max_ppr if max_ppr > 0 else np.nan
    result["Disparate Impact"] = round(float(di), 4)

    # ── Demographic Parity Difference ─────────────────────────────────────
    result["Dem. Parity Diff"] = round(float(max_ppr - min_ppr), 4)

    # ── Equalized Odds Diff (TPR gap) ─────────────────────────────────────
    tprs = result["TPR (Recall)"].values
    eq_odds_diff = round(float(np.nanmax(tprs) - np.nanmin(tprs)), 4)
    result["Eq. Odds Diff (TPR)"] = eq_odds_diff

    return result


def fairness_summary(metrics_df):
    """Return a human-readable list of (label, detail) fairness verdicts."""
    di      = metrics_df["Disparate Impact"].iloc[0]
    dp_diff = metrics_df["Dem. Parity Diff"].iloc[0]
    eq_diff = metrics_df["Eq. Odds Diff (TPR)"].iloc[0]

    verdicts = []
    if not np.isnan(di):
        if di >= 0.8:
            verdicts.append(("PASS Disparate Impact", f"{di:.3f} >= 0.80 -> FAIR"))
        else:
            verdicts.append(("WARN Disparate Impact", f"{di:.3f} < 0.80 -> BIASED"))

    if dp_diff <= 0.10:
        verdicts.append(("PASS Demographic Parity", f"Diff = {dp_diff:.3f} <= 0.10 -> FAIR"))
    else:
        verdicts.append(("WARN Demographic Parity", f"Diff = {dp_diff:.3f} > 0.10 -> BIASED"))

    if eq_diff <= 0.10:
        verdicts.append(("PASS Equalized Odds", f"TPR Gap = {eq_diff:.3f} <= 0.10 -> FAIR"))
    else:
        verdicts.append(("WARN Equalized Odds", f"TPR Gap = {eq_diff:.3f} > 0.10 -> BIASED"))

    return verdicts


def reweigh_samples(df, sensitive_col, label_col):
    """
    Reweighing bias mitigation: compute sample weights so each
    (group, label) combination has equal expected representation.

    Returns np.array of sample weights.
    """
    n = len(df)
    weights = np.ones(n)
    groups = df[sensitive_col].unique()
    labels = df[label_col].unique()

    for grp in groups:
        for lbl in labels:
            mask = (df[sensitive_col] == grp) & (df[label_col] == lbl)
            n_gl = mask.sum()
            n_g  = (df[sensitive_col] == grp).sum()
            n_l  = (df[label_col] == lbl).sum()
            if n_gl > 0:
                expected = (n_g / n) * (n_l / n)
                observed = n_gl / n
                weights[mask] = expected / observed

    return weights


def threshold_optimization(y_prob, y_true, sensitive_col, df, metric="equalized_odds"):
    """
    Find per-group decision thresholds that optimise a fairness metric.
    Returns dict: {group_value: threshold}
    """
    from sklearn.metrics import roc_curve

    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    groups = df[sensitive_col].unique()
    thresholds = {}

    for grp in groups:
        mask = (df[sensitive_col] == grp).values
        if mask.sum() == 0:
            thresholds[grp] = 0.5
            continue
        fpr, tpr, thresh = roc_curve(y_true[mask], y_prob[mask])
        # Pick threshold that maximises TPR - FPR (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        thresholds[grp] = round(float(thresh[best_idx]), 4)

    return thresholds

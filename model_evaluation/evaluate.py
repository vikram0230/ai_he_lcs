import numpy as np
import pandas as pd
from numbers import Number
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

def generate_results(model, X, y, ax_pr, ax_roc, z_index=0,
    plot_label='Line', plot_color='#000000', draw_roc_diagonal=False,
    n_digits=3):
    # PR Curve
    _x, _y = get_curve(model, X, y, curve='pr')
    pr_auc = auc(_x, _y)
    ax_pr.plot(_x, _y, color=plot_color,
        label=plot_label + f' AUC {pr_auc:.2f}',
        zorder=z_index)
    baseline = (y.sum() / len(y))
    ax_pr.plot([0,1], [baseline, baseline], color=plot_color + '80', linestyle='dashed', zorder=z_index-6)
    ax_pr.legend(loc='upper right', fontsize=7, frameon=False)

    # ROC Curve
    _x, _y = get_curve(model, X, y, curve='roc')
    roc_auc = auc(_x, _y)
    ax_roc.plot(_x, _y, color=plot_color,
        label=plot_label + f' AUC {roc_auc:.2f}',
        zorder=z_index)
    if draw_roc_diagonal:
        ax_roc.plot([0,1], [0,1], color='#E7E7E7', linestyle='dashed', zorder=z_index-6)
    ax_roc.legend(loc='lower right', fontsize=7, frameon=False)

    # Confusion Matrix
    sen, spe, ppv, npv = get_confusion_matrix(model, X, y)
    cols = ['label', 'roc_auc', 'pr_auc', 'sensitivity', 'specificity', 'ppv', 'npv',
        'proportion', 'n_diagnosed', 'n_total']
    output = [plot_label, roc_auc, pr_auc, sen, spe, ppv, npv, baseline, y.sum(), len(y)]
    for i in range(len(output)):
        if isinstance(output[i], Number):
            output[i] = round(output[i], n_digits)
    return pd.DataFrame([output], columns=cols)
    # return pandas dataframe of all the table deta points.

def get_curve(model, X, y, curve='roc', decision=0.5): # curve = 'roc' or 'pr'
    y = y.astype(int)
    pred_y = model(X)
    if len(pred_y.shape) > 1:
        pred_y = [p[pred_y.shape[1]-1] for p in pred_y]
    if curve == 'pr':
        precision, recall, _ = precision_recall_curve(y, pred_y)
        return recall, precision
    elif curve == 'roc':
        fpr, tpr, _ = roc_curve(y, pred_y)
        return fpr, tpr
    else:
        return None, None
    
def get_confusion_matrix(model, X, y, sensitivity=0.8):
    y = y.astype(int)
    pred_y = model(X)
    if len(pred_y.shape) > 1:
        pred_y = [p[pred_y.shape[1]-1] for p in pred_y]

    _, tpr, thresholds = roc_curve(y, pred_y)
    cutoff = thresholds[np.abs(tpr - sensitivity).argmin()]
    decision = [1 if prob >= cutoff else 0 for prob in pred_y]

    tn, fp, fn, tp = confusion_matrix(y, decision).ravel()
    sen = tp/(tp+fn) if (tp+fn) != 0 else np.nan
    spe = tn/(tn+fp) if (tn+fp) != 0 else np.nan
    ppv = tp/(tp+fp) if (tp+fp) != 0 else np.nan
    npv = tn/(tn+fn) if (tn+fn) != 0 else np.nan

    return sen, spe, ppv, npv
import numpy as np
import pandas as pd
import scipy.stats as st
from numbers import Number
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

def generate_results(model, X, y, ax_pr, ax_roc, z_index=0,
    plot_label='Line', plot_color='#000000', draw_roc_diagonal=False,
    n_digits=3, verbose=False
):
    # PR Curve
    _x, _y = _get_curve(model, X, y, curve='pr', verbose=verbose)
    pr_auc = auc(_x, _y)
    ax_pr.plot(_x, _y, color=plot_color,
        label=plot_label + f' AUC {pr_auc:.2f}',
        zorder=z_index)
    baseline = (y.sum() / len(y))
    ax_pr.plot([0,1], [baseline, baseline], color=plot_color + '80', linestyle='dashed', zorder=z_index-6)
    ax_pr.legend(loc='upper right', fontsize=7, frameon=False)

    # ROC Curve
    _x, _y = _get_curve(model, X, y, curve='roc', verbose=verbose)
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

def generate_results_ci(model, X_list, y, ax_pr, ax_roc, z_index=0,
    plot_label='Line', plot_color='#000000', draw_roc_diagonal=False,
    n_digits=3, verbose=False, n_points=None, confidence=0.95
):
    # like 'generate_results' above, but with CIs.

    # PR Curve
    _xs, _ys = [], []
    for i in range(len(X_list)):
        _x, _y = _get_curve(model, X_list[i], y, curve='pr',
            verbose=verbose, n_points=n_points)
        _xs.append(_x)
        _ys.append(_y)
    _x = np.mean(np.stack(_xs, axis=0), axis=0)
    _y = np.mean(np.stack(_ys, axis=0), axis=0)
    intervals = []
    for i in range(len(_ys[0])):
        values = [arr[i] for arr in _ys]
        interval = st.t.interval(confidence, len(values)-1, loc=np.mean(values), scale=st.sem(values))
        intervals.append(interval)
    
    pr_auc = auc(_x, _y)
    ax_pr.plot(_x, _y, color=plot_color,
        label=plot_label + f' AUC {pr_auc:.2f}',
        zorder=z_index)
    baseline = (y.sum() / len(y))
    ax_pr.plot([0,1], [baseline, baseline], color=plot_color + '80', linestyle='dashed', zorder=z_index-6)
    ax_pr.legend(loc='upper right', fontsize=7, frameon=False)

    ax_pr.fill_between(
        _x,
        [i[0] for i in intervals], [i[1] for i in intervals],
        color=plot_color, alpha=.2
    )

    # ROC Curve
    _xs, _ys = [], []
    for i in range(len(X_list)):
        _x, _y = _get_curve(model, X_list[i], y, curve='roc',
            verbose=verbose, n_points=n_points)
        _xs.append(_x)
        _ys.append(_y)
    _x = np.mean(np.stack(_xs, axis=0), axis=0)
    _y = np.mean(np.stack(_ys, axis=0), axis=0)
    intervals = []
    for i in range(len(_ys[0])):
        values = [arr[i] for arr in _ys]
        interval = st.t.interval(confidence, len(values)-1, loc=np.mean(values), scale=st.sem(values))
        intervals.append(interval)

    roc_auc = auc(_x, _y)
    ax_roc.plot(_x, _y, color=plot_color,
        label=plot_label + f' AUC {roc_auc:.2f}',
        zorder=z_index)
    if draw_roc_diagonal:
        ax_roc.plot([0,1], [0,1], color='#E7E7E7', linestyle='dashed', zorder=z_index-6)
    ax_roc.legend(loc='lower right', fontsize=7, frameon=False)

    ax_roc.fill_between(
        _x,
        [i[0] for i in intervals], [i[1] for i in intervals],
        color=plot_color, alpha=.2
    )

    # Confusion Matrix
    sens, spes, ppvs, npvs = [], [], [], []
    for i in range(len(X_list)):
        sen, spe, ppv, npv = get_confusion_matrix(model, X_list[i], y)
        sens.append(sen)
        spes.append(spe)
        ppvs.append(ppv)
        npvs.append(npv)
    sen, spe, ppv, npv = np.mean(sens), np.mean(spes), np.mean(ppvs), np.mean(npvs)

    cols = ['label', 'roc_auc', 'pr_auc', 'sensitivity', 'specificity', 'ppv', 'npv',
        'proportion', 'n_diagnosed', 'n_total']
    output = [plot_label, roc_auc, pr_auc, sen, spe, ppv, npv, baseline, y.sum(), len(y)]
    for i in range(len(output)):
        if isinstance(output[i], Number):
            output[i] = round(output[i], n_digits)
    return pd.DataFrame([output], columns=cols)
    # return pandas dataframe of all the table deta points.

def plot_ci_curve(arrays, ax, baseline=None, roc_diagonal=None, confidence=0.95, plot_color='red',
    plot_label='line', legend_position='lower right', layer=1
):
    mean_x_val = np.linspace(0,1,len(arrays[0]))
    means = np.mean(np.stack(arrays, axis=0), axis=0)
    intervals = []
    for i in range(len(arrays[0])):
        values = [arr[i] for arr in arrays]
        interval = st.t.interval(confidence, len(values)-1, loc=np.mean(values), scale=st.sem(values))
        intervals.append(interval)
    if baseline:
        ax.plot([0,1], [baseline, baseline], color=plot_color + '80', linestyle='dashed', zorder=layer)
        # ax.text(0, baseline + 0.02, f'Baseline = {baseline}', color='gray')
    if roc_diagonal:
        ax.plot([0,1], [0,1], color=roc_diagonal, linestyle='dashed')
    area_under_curve = auc(mean_x_val, means)
    ax.plot(mean_x_val, means, color=plot_color,
        label=plot_label + f' AUC {area_under_curve:.2f}',
        zorder=layer)
    ax.fill_between(
        mean_x_val,
        [i[0] for i in intervals], [i[1] for i in intervals],
        color=plot_color, alpha=.2
    )
    ax.legend(loc=legend_position, fontsize=7)

def _get_curve(model, X, y, curve='roc', verbose=False, n_points=None): # curve = 'roc' or 'pr'
    y = y.astype(int)
    pred_y = model(X)
    if len(pred_y.shape) > 1:
        pred_y = [p[pred_y.shape[1]-1] for p in pred_y]
    if curve == 'pr':
        precision, recall, _ = precision_recall_curve(y, pred_y)
        if n_points is not None:
            x_out = np.linspace(0,1,n_points)
            y_out = np.interp(x_out, np.flip(recall), np.flip(precision))
            return x_out, y_out
        return recall, precision
    elif curve == 'roc':
        fpr, tpr, thresholds = roc_curve(y, pred_y)
        if verbose:
            data = []
            for sensitivity in [i/10.0 for i in range(1,10)]:
                cutoff = thresholds[np.abs(tpr - sensitivity).argmin()]
                decision = [1 if prob >= cutoff else 0 for prob in pred_y]
                _, fp, fn, tp = confusion_matrix(y, decision).ravel()
                sen = tp/(tp+fn) if (tp+fn) != 0 else np.nan
                ppv = tp/(tp+fp) if (tp+fp) != 0 else np.nan
                current_data = [sen, ppv]
                for i in range(len(current_data)):
                    current_data[i] = round(current_data[i], 3)
                data.append(current_data)
            print(pd.DataFrame(data, columns=['sensitivity', 'PPV']))
        if n_points is not None:
            x_out = np.linspace(0,1,n_points)
            y_out = np.interp(x_out, fpr, tpr)
            return x_out, y_out
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
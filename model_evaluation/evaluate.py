from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from os import getcwd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as font_manager

font_path = 'C:\Windows\Fonts\segoeui.ttf'
prop = font_manager.FontProperties(fname=font_path)
prop.set_weight = 'medium'
rcParams['font.family'] = prop.get_name()
rcParams['font.weight'] = 'medium'

rcParams['figure.dpi'] = 300

def generate_figure(models, X, y,
    output=getcwd() + '/figure.svg'):
    f = plt.figure(figsize=(6.5,3), dpi=144)
    ax_pr = f.add_subplot(121)
    ax_pr.set_xlabel("Recall", labelpad=-10)
    ax_pr.set_ylabel("Precision", labelpad=-10)
    ax_pr.set_xlim(-0.05, 1.05)
    ax_pr.set_ylim(-0.05, 1.05)
    ax_pr.set_xticks([0,1])
    ax_pr.set_yticks([0,1])

    ax_roc = f.add_subplot(122)
    ax_roc.set_xlabel("False positive rate", labelpad=-10)
    ax_roc.set_ylabel("True positive rate", labelpad=-10)
    ax_roc.set_xlim(-0.05, 1.05)
    ax_roc.set_ylim(-0.05, 1.05)
    ax_roc.set_xticks([0,1])
    ax_roc.set_yticks([0,1])

    print("Generating Figure!")
    for name, model in models.items():
        print(name)
        _x, _y = get_curve(model, X, y, curve='pr')
        ax_pr.plot(_x, _y)
        _x, _y = get_curve(model, X, y, curve='roc')
        ax_roc.plot(_x, _y)
    f.savefig(output, format='svg', bbox_inches='tight')

def get_curve(model, X, y, curve='roc'): # curve = 'roc' or 'pr'
    y = y.astype(int)
    pred_y = model(X)
    if len(pred_y.shape) > 1:
        pred_y = [p[pred_y.shape[1]-1] for p in pred_y]
    if curve == 'pr':
        precision, recall, _ = precision_recall_curve(y, pred_y)
        print(auc(recall, precision))
        return recall, precision
    elif curve == 'roc':
        fpr, tpr, _ = roc_curve(y, pred_y)
        print(auc(fpr, tpr))
        return fpr, tpr
    else:
        return None, None

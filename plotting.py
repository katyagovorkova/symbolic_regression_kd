import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

def get_metric(background_loss, bsm_loss):

    target_val = np.concatenate((np.ones(bsm_loss.shape[0]), np.zeros(background_loss.shape[0])))
    predicted_val = np.concatenate((bsm_loss, background_loss))

    fpr, tpr, threshold_loss = roc_curve(target_val, predicted_val)
    auc_val = auc(fpr, tpr)

    return fpr, tpr, auc_val

def plot_rocs(background_loss, bsm_loss, bsm_name, color, linestyle=None, alpha=1):

    fpr, tpr, auc = get_metric(background_loss, bsm_loss)
    plt.plot(
        fpr, tpr,
        '-',
        label=f'{bsm_name} (AUC = {auc*100:.0f}%)',
        linewidth=3,
        linestyle=linestyle,
        color=color,
        alpha=alpha)

    plt.xlim(10**(-6),1)
    plt.ylim(10**(-6),1.2)
    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=3, linestyle='--')
    plt.vlines(1e-5, 0, 1, linestyles='--', color='#ef5675', linewidth=3)
    plt.legend(loc='lower right', frameon=False, title=f'ROC {bsm_name}')
    plt.tight_layout()
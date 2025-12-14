import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_score_histograms(scores_member, scores_non_member, title="Score Distribution", save_path=None):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(scores_member, fill=True, label='Member', color='blue', alpha=0.3)
    sns.kdeplot(scores_non_member, fill=True, label='Non-Member', color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel('MIA Score')
    plt.ylabel('Density')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_generalization_vs_attack(gaps, aucs, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(gaps, aucs, color='purple', s=100)
    plt.xlabel('Generalization Gap (Train Acc - Test Acc)')
    plt.ylabel('Attack AUC')
    plt.title('Generalization Gap vs Membership Inference Risk')
    plt.grid(True)
    
    # Add trend line
    if len(gaps) > 1:
        z = np.polyfit(gaps, aucs, 1)
        p = np.poly1d(z)
        plt.plot(gaps, p(gaps), "r--")
        
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

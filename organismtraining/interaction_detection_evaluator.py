import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


class PubmedProteinInteractionEvaluator:
    def evaluate(self, predicted_labels: list, true_labels: list):
        fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, predicted_labels)

        sns.set(style='whitegrid')
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
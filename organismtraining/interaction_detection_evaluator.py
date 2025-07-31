import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from organismtraining.interaction_detection_model import PubmedProteinInteractionTrainer
import argparse

def _round(x, threshold=0.9):
    return 1 if x >= threshold else 0

def _manual(y_scores, y_test):
    y_scores = np.array(y_scores)
    y_test = np.array(y_test)
    thresholds = np.sort(np.unique(y_scores))
    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)

        TP = np.sum((y_test == 1) & (y_pred == 1))
        FP = np.sum((y_test == 0) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    # Sort by recall for plotting and AUPRC calculation
    auprc_manual = auc(recalls, precisions)

    return recalls, precisions, auprc_manual, thresholds

class PubmedProteinInteractionEvaluator:
    def evaluate(self, predicted_labels: list, true_labels: list):
        fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, predicted_labels)

        sns.set_style('whitegrid', {'axes.grid' : False})
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')

        recall, precision, auprc, thresholds = _manual(predicted_labels, true_labels)
        sns.set_style('whitegrid', {'axes.grid' : False})
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14, fontname='Arial')
        plt.ylabel('Precision', fontsize=14, fontname='Arial')
        plt.xticks(fontname='Arial')
        plt.yticks(fontname='Arial')
        plt.savefig('prc_curve.pdf', format="pdf")

        cm = confusion_matrix(true_labels, [_round(x) for x in predicted_labels])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        for text in ax.texts:  # Includes number labels
            text.set_fontname('Arial')
            text.set_fontsize(12)

        # Change font for axis labels and title
        ax.set_xlabel("Predicted label", fontname='Arial', fontsize=14)
        ax.set_ylabel("True label", fontname='Arial', fontsize=14)
        plt.savefig('cf_9.pdf', format="pdf")

        cm = confusion_matrix(true_labels, [_round(x, threshold=0.7) for x in predicted_labels])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig('cf_7.png')

        cm = confusion_matrix(true_labels, [_round(x, threshold=0.5) for x in predicted_labels])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig('cf_5.png')


    def chart(self, predicted_labels: list):
        binwidth = 0.01
        sns.set_style('whitegrid', {'axes.grid' : False})
        plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots()
        ax.set_ylim(0, 55000)
        plt.hist(predicted_labels, edgecolor='black', bins=[float(i) / 100 for i in range(101)])
        plt.xlabel('Predicted Labels', fontname='Arial')
        plt.ylabel('Number of Abstracts', fontname='Arial')
        plt.xticks(fontname='Arial')
        plt.yticks(fontname='Arial')
        plt.savefig('neg_predicted_labels_histogram.pdf', format='pdf')

if __name__ == "__main__":
    evaluator = PubmedProteinInteractionEvaluator()

    parser = argparse.ArgumentParser(description='Train and evaluate protein interaction detection model')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the dataset directory containing train.tsv, dev.tsv, and test.tsv files')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model directory containing pretrained_dir')
    parser.add_argument('--retrain', action='store_true',
                      help='Whether to retrain the model from scratch')
    parser.add_argument('--predict_set_path', type=str, required=False,
                      help='Set on which to evaluate the model')
    parser.add_argument('--save_output_path', type=str, required=False,
                      help='Path to save model predictions to')
    parser.add_argument('--from_prediction_csv', type=str, required=False,
                      help='Evaluate on prediction CSV')
    args = parser.parse_args()
    if args.from_prediction_csv:
        df = pd.read_csv(args.from_prediction_csv)
        if 'actual' in df.columns:
            evaluator.evaluate(df['predicted'].tolist(), df['actual'].tolist())
        else:
            evaluator.chart(df['predicted'].tolist())
    else:
        save_output_path = args.save_output_path if args.save_output_path else None

        if args.retrain:
            pubmed_trainer = PubmedProteinInteractionTrainer(dataset_path=args.dataset_path, model_path=args.model_path, save_output_path=save_output_path)
            pubmed_trainer.train(args.model_path)
        else:
            pubmed_trainer = PubmedProteinInteractionTrainer(dataset_path=args.dataset_path, model_path=args.model_path, load_model=True, save_output_path=save_output_path)

        if args.predict_set_path:
            predicted_labels = pubmed_trainer.predict(args.predict_set_path)
            evaluator.chart(predicted_labels)
        else:
            predicted_labels, true_labels = pubmed_trainer.eval_test()
            evaluator.evaluate(predicted_labels, true_labels)
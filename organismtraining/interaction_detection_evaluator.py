import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from organismtraining.interaction_detection_model import PubmedProteinInteractionTrainer
import argparse


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
        plt.savefig('roc_curve.png')

    def chart(self, predicted_labels: list):
        sns.set(style='whitegrid')
        plt.figure(figsize=(8, 6))
        plt.hist(predicted_labels, edgecolor='black')
        plt.xlabel('Predicted Labels')
        plt.ylabel('Frequency')
        plt.title('Predicted Labels Histogram')
        plt.savefig('predicted_labels_histogram.png')

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
    args = parser.parse_args()


    if args.retrain:
        pubmed_trainer = PubmedProteinInteractionTrainer(dataset_path=args.dataset_path, model_path=args.model_path)
        pubmed_trainer.train(args.model_path)
    else:
        pubmed_trainer = PubmedProteinInteractionTrainer(dataset_path=args.dataset_path, model_path=args.model_path, load_model=True)

    if args.predict_set_path:
        predicted_labels = pubmed_trainer.predict(args.predict_set_path)
        evaluator.chart(predicted_labels)
    else:
        predicted_labels, true_labels = pubmed_trainer.eval_test()
        evaluator.evaluate(predicted_labels, true_labels)
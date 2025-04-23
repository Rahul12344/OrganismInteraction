import os
import logging
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_METRIC = evaluate.load("f1")
_PRETRAIN_DIR = "bluebert_pretrained_model"  # Default to base BERT model
_MAX_LENGTH = 512
_SEED = 42

def _tokenize_function(samples: pd.DataFrame, tokenizer: BertTokenizer):
    return tokenizer(samples["text"], padding="max_length", truncation=True, max_length=_MAX_LENGTH)

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = _METRIC.compute(predictions=predictions, references=labels)

    # Add additional metrics
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    probs = _sigmoid(logits[:, 1])  # Get probabilities for positive class
    metrics["auc"] = roc_auc_score(labels, probs)
    metrics["precision"] = precision_score(labels, predictions)
    metrics["recall"] = recall_score(labels, predictions)

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def _sigmoid(x):
    return 1/(1 + np.exp(-x))

class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Add dropout after BERT
        self.dropout = nn.Dropout(0.3)
        # Add intermediate layer
        self.intermediate = nn.Linear(config.hidden_size, 256)
        self.activation = nn.ReLU()
        # Add final classification layer
        self.classifier = nn.Linear(256, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        intermediate_output = self.intermediate(pooled_output)
        intermediate_output = self.activation(intermediate_output)
        logits = self.classifier(intermediate_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(logits.device))  # Class weights
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class PubmedProteinInteractionTrainer:
    def __init__(self, dataset_path: str, model_path: str):
        """Trainer class for detecting virus-protein interactions in Pubmed abstracts."""
        self._tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, _PRETRAIN_DIR))
        self._tokenized_dataset = self._build_tokenized_dataset(dataset_path)
        self._pretrained_model = self._load_model_from_checkpoint(model_path)
        self._trainer = self._build_trainer()

    def train(self):
        """Trains model using train/eval data"""
        logger.info("Starting training...")
        self._trainer.train()
        logger.info("Training completed")

    def predict(self) -> tuple:
        """Performs prediction on test data and returns predicted and actual labels."""
        logger.info("Running predictions on test set...")
        predictions = self._trainer.predict(self._tokenized_dataset["test"])
        true_labels = predictions.label_ids
        predicted_labels = [_sigmoid(prediction[1]) for prediction in predictions.predictions]

        # Log prediction statistics
        logger.info(f"Number of samples: {len(predicted_labels)}")
        logger.info(f"Prediction range: [{min(predicted_labels):.4f}, {max(predicted_labels):.4f}]")
        logger.info(f"Mean prediction: {np.mean(predicted_labels):.4f}")
        logger.info(f"True labels distribution: {np.bincount(true_labels)}")

        return predicted_labels, true_labels

    def _build_trainer(self) -> Trainer:
        # Calculate number of training steps based on dataset size
        num_training_steps = len(self._tokenized_dataset["train"]) * 10  # epochs * dataset size

        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            learning_rate=2e-5,
            logging_steps=100,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        optimizer = Adam(
            params=self._pretrained_model.parameters(),
            lr=2e-5,
            eps=1e-08,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        trainer = Trainer(
            model=self._pretrained_model,
            args=training_args,
            train_dataset=self._tokenized_dataset["train"].shuffle(seed=_SEED),
            eval_dataset=self._tokenized_dataset["dev"].shuffle(seed=_SEED),
            compute_metrics=_compute_metrics,
            optimizers=(optimizer, scheduler),
        )
        return trainer

    def _load_model_from_checkpoint(self, model_path: str) -> BertForSequenceClassification:
        model = BertForSequenceClassification.from_pretrained(
            os.path.join(model_path, _PRETRAIN_DIR),
            num_labels=2
        )
        return model

    def _build_tokenized_dataset(self, dataset_path: str):
        try:
            virus_train = self._to_dataset(
                pd.read_csv(os.path.join(dataset_path, "train.tsv"), sep='\t')
            )
            virus_dev = self._to_dataset(
                pd.read_csv(os.path.join(dataset_path, "dev.tsv"), sep='\t')
            )
            virus_test = self._to_dataset(
                pd.read_csv(os.path.join(dataset_path, "test.tsv"), sep='\t')
            )
            dataset = self._build_dataset_dict(
                [
                    ("train", virus_train),
                    ("dev", virus_dev),
                    ("test", virus_test)
                ]
            )
            tokenized_dataset = dataset.map(
                lambda x: _tokenize_function(x, self._tokenizer),
                batched=True
            )
            return tokenized_dataset
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find dataset files in {dataset_path}. Please ensure train.tsv, dev.tsv, and test.tsv exist.") from e
        except Exception as e:
            raise Exception(f"Error building dataset: {str(e)}") from e

    def _build_dataset_dict(self, labeled_datasets: list[tuple[str, Dataset]]) -> DatasetDict:
        return DatasetDict(
            {
                label: dataset for label, dataset in labeled_datasets
            }
        )

    def _to_dataset(self, data: pd.DataFrame) -> Dataset:
        return Dataset.from_dict({
            'id': data['abstract'].tolist(),
            'label': data['label'].tolist(),
            'text': data['text'].tolist()
        })
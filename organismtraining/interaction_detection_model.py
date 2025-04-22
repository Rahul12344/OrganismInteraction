import os

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup
from torch.optim import Adam

_METRIC = evaluate.load("f1")
_PRETRAIN_DIR = "bluebert_pretrained_model"  # Default to base BERT model
_MAX_LENGTH = 512
_SEED = 42

def _tokenize_function(samples: pd.DataFrame, tokenizer: BertTokenizer):
    return tokenizer(samples["text"], padding="max_length", truncation=True, max_length=_MAX_LENGTH)

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return _METRIC.compute(predictions=predictions, references=labels)

def _sigmoid(x):
    return 1/(1 + np.exp(-x))

class PubmedProteinInteractionTrainer:
    def __init__(self, dataset_path: str, model_path: str):
        """Trainer class for detecting virus-protein interactions in Pubmed abstracts."""
        self._tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, _PRETRAIN_DIR))
        self._tokenized_dataset = self._build_tokenized_dataset(dataset_path)
        self._pretrained_model = self._load_model_from_checkpoint(model_path)
        self._trainer = self._build_trainer()

    def train(self):
        """Trains model using train/eval data"""
        self._trainer.train()

    def predict(self) -> tuple:
        """Performs prediction on test data and returns predicted and actual labels."""
        predictions = self._trainer.predict(self._tokenized_dataset["test"])
        true_labels = predictions.label_ids
        predicted_labels = [_sigmoid(prediction[1]) for prediction in predictions.predictions]
        return predicted_labels, true_labels


    def _build_trainer(self) -> Trainer:
        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5
        )
        optimizer = Adam(
            params=self._pretrained_model.parameters(),
            lr=1e-05,
            eps=1e-08
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=1668
        )
        trainer = Trainer(
            model=self._pretrained_model,
            args=training_args,
            train_dataset=self._tokenized_dataset["train"].shuffle(seed=_SEED),
            eval_dataset=self._tokenized_dataset["dev"].shuffle(seed=_SEED),
            compute_metrics=_compute_metrics,
            optimizers=(optimizer,scheduler),
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
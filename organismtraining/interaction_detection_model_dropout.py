import os
import logging
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
import torch
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_METRIC = evaluate.load("f1")
_PRETRAIN_DIR = "bluebert_pretrained_model"
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
        # Initialize with parent class but don't create the classifier yet
        super(BertForSequenceClassification, self).__init__(config)

        # Get the BERT model
        self.bert = self.bert

        # Add dropout after BERT
        self.dropout = nn.Dropout(0.3)

        # Add intermediate layer
        self.intermediate = nn.Linear(config.hidden_size, 256)
        self.activation = nn.ReLU()

        # Add final classification layer
        self.classifier = nn.Linear(256, config.num_labels)

        # Initialize weights for new layers
        self._init_weights(self.intermediate)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        # Get BERT outputs
        outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds)

        # Get pooled output
        pooled_output = outputs[1]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Pass through intermediate layer
        intermediate_output = self.intermediate(pooled_output)
        intermediate_output = self.activation(intermediate_output)

        # Get logits
        logits = self.classifier(intermediate_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            # Calculate class weights based on label distribution
            if hasattr(self, 'class_weights'):
                weight = self.class_weights
            else:
                weight = torch.tensor([1.0, 2.0]).to(logits.device)

            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def set_class_weights(self, class_weights):
        """Set class weights for loss calculation"""
        self.class_weights = torch.tensor(class_weights).float()

class PubmedProteinInteractionTrainer:
    def __init__(self, dataset_path: str, model_path: str, tune_hyperparams: bool = False):
        """Trainer class for detecting virus-protein interactions in Pubmed abstracts."""
        logger.info(f"Initializing trainer with model from {model_path}")
        self._tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, _PRETRAIN_DIR))
        self._tokenized_dataset = self._build_tokenized_dataset(dataset_path)
        self._pretrained_model = self._load_model_from_checkpoint(model_path)
        self._tune_hyperparams = tune_hyperparams

        # Calculate and set class weights
        train_labels = self._tokenized_dataset["train"]["label"]
        label_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        class_weights = [total_samples / (len(label_counts) * count) for count in label_counts]
        logger.info(f"Using class weights: {class_weights}")
        self._pretrained_model.set_class_weights(class_weights)

        if tune_hyperparams:
            self._tune_hyperparameters()
        else:
            self._trainer = self._build_trainer()

    def _tune_hyperparameters(self):
        """Tune hyperparameters using k-fold cross validation."""
        from .hyperparameter_tuning import HyperparameterTuner

        # Combine train and dev sets for cross-validation
        full_train_dataset = self._tokenized_dataset["train"].concatenate(self._tokenized_dataset["dev"])

        # Initialize tuner
        tuner = HyperparameterTuner(
            model_class=CustomBertForSequenceClassification,
            tokenizer=self._tokenizer,
            dataset=full_train_dataset,
            n_splits=5
        )

        # Run hyperparameter tuning
        best_params = tuner.tune()

        # Log best parameters
        logger.info(f"Best hyperparameters found: {best_params}")

        # Create trainer with best parameters
        self._trainer = self._build_trainer_with_params(best_params)

    def _build_trainer_with_params(self, params: Dict[str, Any]) -> Trainer:
        """Build trainer with specific parameters."""
        num_training_steps = len(self._tokenized_dataset["train"]) * params['num_epochs']
        num_warmup_steps = num_training_steps // 10

        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            per_device_train_batch_size=params['batch_size'],
            per_device_eval_batch_size=params['batch_size'],
            num_train_epochs=params['num_epochs'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            warmup_steps=num_warmup_steps,
            gradient_accumulation_steps=4,
            fp16=True,
            logging_steps=100,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="auc"
        )

        optimizer = Adam(
            params=self._pretrained_model.parameters(),
            lr=params['learning_rate'],
            eps=1e-08,
            weight_decay=params['weight_decay']
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Set dropout rate
        self._pretrained_model.dropout.p = params['dropout_rate']

        return Trainer(
            model=self._pretrained_model,
            args=training_args,
            train_dataset=self._tokenized_dataset["train"].shuffle(seed=_SEED),
            eval_dataset=self._tokenized_dataset["dev"].shuffle(seed=_SEED),
            compute_metrics=_compute_metrics,
            optimizers=(optimizer, scheduler),
        )

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
        num_warmup_steps = num_training_steps // 10  # 10% warmup
        logger.info(f"Number of training steps: {num_training_steps}")
        logger.info(f"Number of warmup steps: {num_warmup_steps}")

        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=num_warmup_steps,
            gradient_accumulation_steps=4,
            fp16=True,  # Enable mixed precision training
            logging_steps=100,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="auc"
        )

        optimizer = Adam(
            params=self._pretrained_model.parameters(),
            lr=2e-5,
            eps=1e-08,
            weight_decay=0.01
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
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
        config = BertConfig.from_pretrained(
            os.path.join(model_path, _PRETRAIN_DIR),
            num_labels=2,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3
        )

        model = CustomBertForSequenceClassification.from_pretrained(
            os.path.join(model_path, _PRETRAIN_DIR),
            config=config,
            ignore_mismatched_sizes=True  # Add this to handle the custom architecture
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

            # Log dataset statistics
            logger.info(f"Train set size: {len(virus_train)}")
            logger.info(f"Dev set size: {len(virus_dev)}")
            logger.info(f"Test set size: {len(virus_test)}")

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
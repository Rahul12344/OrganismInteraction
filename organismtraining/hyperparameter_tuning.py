import numpy as np
from sklearn.model_selection import KFold
from transformers import TrainingArguments, Trainer
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
import logging
from typing import Dict, List, Tuple, Any
import torch
from .interaction_detection_model_dropout import CustomBertForSequenceClassification, _compute_metrics

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self,
                 model_class: CustomBertForSequenceClassification,
                 tokenizer,
                 dataset,
                 n_splits: int = 5,
                 param_grid: Dict[str, List[Any]] = None):
        """
        Initialize the hyperparameter tuner.

        Args:
            model_class: The model class to use
            tokenizer: The tokenizer to use
            dataset: The full dataset to split
            n_splits: Number of folds for cross-validation
            param_grid: Dictionary of hyperparameters to tune
        """
        self.model_class = model_class
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Default parameter grid if none provided
        self.param_grid = param_grid or {
            'learning_rate': [1e-5, 2e-5, 5e-5],
            'batch_size': [8, 16, 32],
            'num_epochs': [5, 10, 15],
            'dropout_rate': [0.1, 0.2, 0.3],
            'weight_decay': [0.01, 0.1, 0.2]
        }

        self.best_params = None
        self.best_score = -np.inf
        self.cv_results = []

    def _create_trainer(self,
                       model: CustomBertForSequenceClassification,
                       train_dataset,
                       eval_dataset,
                       params: Dict[str, Any]) -> Trainer:
        """Create a trainer with given parameters."""
        num_training_steps = len(train_dataset) * params['num_epochs']
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
            params=model.parameters(),
            lr=params['learning_rate'],
            eps=1e-08,
            weight_decay=params['weight_decay']
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=_compute_metrics,
            optimizers=(optimizer, scheduler),
        )

    def _evaluate_params(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Evaluate a set of parameters using k-fold cross validation."""
        fold_scores = []
        fold_metrics = []

        # Get indices for k-fold splitting
        indices = np.arange(len(self.dataset))

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(indices)):
            logger.info(f"Training fold {fold + 1}/{self.n_splits}")

            # Split the dataset
            train_dataset = self.dataset.select(train_idx)
            val_dataset = self.dataset.select(val_idx)

            # Create and train model
            model = self.model_class.from_pretrained(
                self.model_class.config._name_or_path,
                config=self.model_class.config
            )

            # Set dropout rate
            model.dropout.p = params['dropout_rate']

            trainer = self._create_trainer(model, train_dataset, val_dataset, params)
            trainer.train()

            # Evaluate
            eval_results = trainer.evaluate()
            fold_scores.append(eval_results['eval_auc'])
            fold_metrics.append(eval_results)

            logger.info(f"Fold {fold + 1} AUC: {eval_results['eval_auc']:.4f}")

        # Calculate mean score and metrics
        mean_score = np.mean(fold_scores)
        mean_metrics = {
            k: np.mean([m[k] for m in fold_metrics])
            for k in fold_metrics[0].keys()
        }

        return mean_score, mean_metrics

    def tune(self) -> Dict[str, Any]:
        """Perform grid search over the parameter grid."""
        from itertools import product

        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))

        logger.info(f"Starting grid search with {len(param_combinations)} combinations")

        for i, values in enumerate(param_combinations):
            params = dict(zip(param_names, values))
            logger.info(f"\nEvaluating parameter combination {i + 1}/{len(param_combinations)}")
            logger.info(f"Parameters: {params}")

            score, metrics = self._evaluate_params(params)

            # Store results
            self.cv_results.append({
                'params': params,
                'mean_score': score,
                'metrics': metrics
            })

            # Update best parameters if needed
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                logger.info(f"New best parameters found! Score: {score:.4f}")

        logger.info(f"\nBest parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")

        return self.best_params

    def get_cv_results(self) -> List[Dict]:
        """Get the cross-validation results."""
        return self.cv_results
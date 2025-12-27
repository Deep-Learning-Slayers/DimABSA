"""
Training Pipeline for DimABSA Models

Provides comprehensive training infrastructure including:
- Training loop with progress tracking
- Evaluation metrics (RMSE, MAE, Pearson correlation)
- Early stopping and model checkpointing
- Learning rate scheduling
- Gradient clipping
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ReduceLROnPlateau
from scipy.stats import pearsonr
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimizer
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    adam_betas: Tuple[float, float] = (0.9, 0.999)

    # Training schedule
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    scheduler_type: str = "linear"  # "linear", "cosine", "plateau"

    # Regularization
    gradient_clip_norm: float = 1.0
    dropout: float = 0.1

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_metric: str = "rmse_va"  # Metric to monitor
    early_stopping_mode: str = "min"  # "min" or "max"

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "outputs/checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10  # Log every N batches
    eval_interval: int = 1  # Evaluate every N epochs


@dataclass
class TrainingState:
    """State tracking during training."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    patience_counter: int = 0
    training_history: List[Dict] = field(default_factory=list)


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for VA predictions.

    Args:
        predictions: Dict with 'valence' and 'arousal' arrays
        labels: Dict with 'valence' and 'arousal' arrays

    Returns:
        Dictionary with computed metrics
    """
    pred_v = predictions['valence']
    pred_a = predictions['arousal']
    gold_v = labels['valence']
    gold_a = labels['arousal']

    n = len(pred_v)

    # RMSE for VA (combined, as per competition)
    # RMSE_VA = sqrt(mean((v_pred - v_gold)^2 + (a_pred - a_gold)^2))
    squared_errors = (pred_v - gold_v) ** 2 + (pred_a - gold_a) ** 2
    rmse_va = math.sqrt(np.mean(squared_errors))

    # Normalized RMSE (divided by max possible distance)
    max_dist = math.sqrt(128)  # sqrt(8^2 + 8^2) for [1,9] range
    rmse_va_norm = rmse_va / max_dist

    # Individual RMSE
    rmse_v = math.sqrt(np.mean((pred_v - gold_v) ** 2))
    rmse_a = math.sqrt(np.mean((pred_a - gold_a) ** 2))

    # MAE
    mae_v = np.mean(np.abs(pred_v - gold_v))
    mae_a = np.mean(np.abs(pred_a - gold_a))
    mae_va = (mae_v + mae_a) / 2

    # Pearson correlation
    pcc_v, pcc_v_pval = pearsonr(pred_v, gold_v)
    pcc_a, pcc_a_pval = pearsonr(pred_a, gold_a)

    return {
        'rmse_va': rmse_va,
        'rmse_va_norm': rmse_va_norm,
        'rmse_v': rmse_v,
        'rmse_a': rmse_a,
        'mae_v': mae_v,
        'mae_a': mae_a,
        'mae_va': mae_va,
        'pcc_v': pcc_v,
        'pcc_a': pcc_a,
        'pcc_v_pval': pcc_v_pval,
        'pcc_a_pval': pcc_a_pval,
        'n_samples': n
    }


class DimABSATrainer:
    """
    Trainer class for DimABSA models.

    Handles:
    - Training loop with optimization
    - Validation and metric computation
    - Early stopping
    - Model checkpointing
    - Training history logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        lexicon_extractor: Optional[Callable] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration
            lexicon_extractor: Optional function to extract lexicon features
                              Should take (texts, aspects) and return tensor
        """
        if config is None:
            config = TrainingConfig()

        self.model = model
        self.config = config
        self.lexicon_extractor = lexicon_extractor
        self.state = TrainingState()

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer and scheduler (set during train())
        self.optimizer = None
        self.scheduler = None

        # Loss function
        self.criterion = nn.MSELoss()

    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        # Separate parameters with and without weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            betas=self.config.adam_betas
        )

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "linear":
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps
            )
        elif self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=2
            )
        else:
            self.scheduler = None

    def _prepare_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Move batch to device and prepare inputs."""
        prepared = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            elif key in ['text', 'aspect']:
                prepared[key] = value  # Keep strings as-is

        # Extract lexicon features if extractor is provided
        if self.lexicon_extractor is not None and 'text' in batch and 'aspect' in batch:
            lexicon_features = self.lexicon_extractor(batch['text'], batch['aspect'])
            prepared['lexicon_features'] = lexicon_features.to(self.device)

        return prepared

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            leave=True
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            prepared = self._prepare_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=prepared['input_ids'],
                attention_mask=prepared['attention_mask'],
                aspect_mask=prepared.get('aspect_mask'),
                lexicon_features=prepared.get('lexicon_features'),
                token_type_ids=prepared.get('token_type_ids')
            )

            # Compute loss
            labels = prepared['labels']
            pred_v = outputs['valence']
            pred_a = outputs['arousal']

            loss_v = self.criterion(pred_v, labels[:, 0])
            loss_a = self.criterion(pred_a, labels[:, 1])
            loss = loss_v + loss_a

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            # Optimizer step
            self.optimizer.step()

            # Scheduler step (for step-based schedulers)
            if self.scheduler is not None and self.config.scheduler_type != "plateau":
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.state.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })

        return {
            'train_loss': total_loss / num_batches,
            'train_loss_total': total_loss
        }

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        desc: str = "Evaluating"
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            eval_loader: Evaluation data loader
            desc: Description for progress bar

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        all_pred_v = []
        all_pred_a = []
        all_gold_v = []
        all_gold_a = []
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(eval_loader, desc=desc, leave=False)

        for batch in progress_bar:
            prepared = self._prepare_batch(batch)

            outputs = self.model(
                input_ids=prepared['input_ids'],
                attention_mask=prepared['attention_mask'],
                aspect_mask=prepared.get('aspect_mask'),
                lexicon_features=prepared.get('lexicon_features'),
                token_type_ids=prepared.get('token_type_ids')
            )

            labels = prepared['labels']
            pred_v = outputs['valence']
            pred_a = outputs['arousal']

            # Compute loss
            loss_v = self.criterion(pred_v, labels[:, 0])
            loss_a = self.criterion(pred_a, labels[:, 1])
            loss = loss_v + loss_a
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            all_pred_v.extend(pred_v.cpu().numpy())
            all_pred_a.extend(pred_a.cpu().numpy())
            all_gold_v.extend(labels[:, 0].cpu().numpy())
            all_gold_a.extend(labels[:, 1].cpu().numpy())

        # Compute metrics
        predictions = {
            'valence': np.array(all_pred_v),
            'arousal': np.array(all_pred_a)
        }
        labels = {
            'valence': np.array(all_gold_v),
            'arousal': np.array(all_gold_a)
        }

        metrics = compute_metrics(predictions, labels)
        metrics['eval_loss'] = total_loss / num_batches

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        model_name: str = "model"
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            eval_loader: Optional validation data loader
            model_name: Name for saving checkpoints

        Returns:
            Training history
        """
        # Initialize optimizer and scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs
        self._create_optimizer()
        self._create_scheduler(num_training_steps)

        # Reset training state
        self.state = TrainingState()

        # Determine early stopping direction
        if self.config.early_stopping_mode == "min":
            self.state.best_metric = float('inf')
            is_better = lambda x, best: x < best
        else:
            self.state.best_metric = float('-inf')
            is_better = lambda x, best: x > best

        print(f"\n{'='*60}")
        print(f"Starting training: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        if eval_loader:
            print(f"  Validation samples: {len(eval_loader.dataset)}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Evaluation
            if eval_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(eval_loader, desc="Validating")

                # Log epoch results
                print(f"\nEpoch {epoch + 1} Results:")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"  Val Loss: {eval_metrics['eval_loss']:.4f}")
                print(f"  Val RMSE_VA: {eval_metrics['rmse_va']:.4f}")
                print(f"  Val PCC_V: {eval_metrics['pcc_v']:.4f}")
                print(f"  Val PCC_A: {eval_metrics['pcc_a']:.4f}")

                # Record history
                epoch_record = {
                    'epoch': epoch + 1,
                    **train_metrics,
                    **{f'val_{k}': v for k, v in eval_metrics.items()}
                }
                self.state.training_history.append(epoch_record)

                # Check for improvement
                current_metric = eval_metrics[self.config.early_stopping_metric]

                if is_better(current_metric, self.state.best_metric):
                    self.state.best_metric = current_metric
                    self.state.best_epoch = epoch + 1
                    self.state.patience_counter = 0

                    # Save best model
                    if self.config.save_best_only:
                        self.save_checkpoint(
                            self.checkpoint_dir / f"{model_name}_best.pt"
                        )
                        print(f"  -> New best model saved! ({self.config.early_stopping_metric}: {current_metric:.4f})")
                else:
                    self.state.patience_counter += 1
                    print(f"  -> No improvement ({self.state.patience_counter}/{self.config.early_stopping_patience})")

                # Update scheduler for plateau
                if self.config.scheduler_type == "plateau" and self.scheduler is not None:
                    self.scheduler.step(current_metric)

                # Early stopping check
                if self.state.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            else:
                # Record training-only history
                self.state.training_history.append({
                    'epoch': epoch + 1,
                    **train_metrics
                })

        training_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Training completed in {training_time:.1f}s")
        print(f"Best epoch: {self.state.best_epoch}")
        print(f"Best {self.config.early_stopping_metric}: {self.state.best_metric:.4f}")
        print(f"{'='*60}\n")

        return {
            'best_metric': self.state.best_metric,
            'best_epoch': self.state.best_epoch,
            'training_time': training_time,
            'history': self.state.training_history
        }

    def save_checkpoint(self, path: Union[str, Path]):
        """Save model checkpoint."""
        path = Path(path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_state': asdict(self.state),
            'config': asdict(self.config)
        }, path)

    def load_checkpoint(self, path: Union[str, Path]):
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'training_state' in checkpoint:
            state_dict = checkpoint['training_state']
            self.state = TrainingState(**{
                k: v for k, v in state_dict.items()
                if k in TrainingState.__dataclass_fields__
            })


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader] = None,
    lexicon_extractor: Optional[Callable] = None,
    config: Optional[TrainingConfig] = None,
    model_name: str = "model"
) -> Dict:
    """
    Convenience function for training a model.

    Args:
        model: Model to train
        train_loader: Training data loader
        eval_loader: Validation data loader
        lexicon_extractor: Function to extract lexicon features
        config: Training configuration
        model_name: Name for checkpoints

    Returns:
        Training results
    """
    trainer = DimABSATrainer(
        model=model,
        config=config,
        lexicon_extractor=lexicon_extractor
    )

    return trainer.train(
        train_loader=train_loader,
        eval_loader=eval_loader,
        model_name=model_name
    )


def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    lexicon_extractor: Optional[Callable] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Convenience function for evaluating a model.

    Args:
        model: Model to evaluate
        eval_loader: Evaluation data loader
        lexicon_extractor: Function to extract lexicon features
        device: Device to use

    Returns:
        Evaluation metrics
    """
    config = TrainingConfig(device=device)
    trainer = DimABSATrainer(
        model=model,
        config=config,
        lexicon_extractor=lexicon_extractor
    )

    return trainer.evaluate(eval_loader)


if __name__ == "__main__":
    # Test metric computation
    print("Testing metric computation...")

    np.random.seed(42)
    n = 100

    # Generate dummy predictions
    gold_v = np.random.uniform(1, 9, n)
    gold_a = np.random.uniform(1, 9, n)
    pred_v = gold_v + np.random.normal(0, 0.5, n)
    pred_a = gold_a + np.random.normal(0, 0.8, n)

    predictions = {'valence': pred_v, 'arousal': pred_a}
    labels = {'valence': gold_v, 'arousal': gold_a}

    metrics = compute_metrics(predictions, labels)

    print("\nComputed metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

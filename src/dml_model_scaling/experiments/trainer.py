"""Model training utilities."""

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dml_model_scaling.exceptions import ModelError
from dml_model_scaling.types import DeviceType
from dml_model_scaling.utils.logging import get_logger

logger = get_logger(__name__)


def get_device(device: DeviceType = "auto") -> torch.device:
    """Get the appropriate device for training.

    Args:
        device: Device specification. Can be 'auto', 'cpu', 'cuda', or 'mps'.

    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


class ModelTrainer:
    """Handles model training and evaluation.

    Attributes:
        model: The neural network model to train.
        device: Device to use for training.
        criterion: Loss function.
        optimizer: Optimizer instance.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: DeviceType = "auto",
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train.
            learning_rate: Learning rate for optimizer.
            device: Device for training.
        """
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        logger.info(f"Initialized trainer on device: {self.device}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, test_loader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            Tuple of (accuracy, average_loss).
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

                total_loss += loss.item()
                num_batches += 1

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return accuracy, avg_loss

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        verbose: bool = True,
    ) -> tuple[float, float]:
        """Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for test data.
            epochs: Number of epochs to train.
            verbose: Whether to print progress.

        Returns:
            Tuple of (final_accuracy, training_time).

        Raises:
            ModelError: If training fails.
        """
        if epochs <= 0:
            raise ModelError(f"epochs must be positive, got {epochs}")

        start_time = time.time()

        try:
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader)

                if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                    accuracy, test_loss = self.evaluate(test_loader)
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Test Loss: {test_loss:.4f}, "
                        f"Accuracy: {accuracy:.4f}"
                    )

            training_time = time.time() - start_time
            final_accuracy, _ = self.evaluate(test_loader)

            return final_accuracy, training_time

        except Exception as e:
            raise ModelError(f"Training failed: {e}") from e


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float = 0.001,
    device: DeviceType = "auto",
) -> tuple[float, float]:
    """Convenience function to train and evaluate a model.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        test_loader: Test data loader.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        device: Device for training.

    Returns:
        Tuple of (accuracy, training_time).
    """
    trainer = ModelTrainer(model, learning_rate=learning_rate, device=device)
    return trainer.train(train_loader, test_loader, epochs, verbose=False)

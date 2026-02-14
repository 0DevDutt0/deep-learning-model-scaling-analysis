"""Base model interface for all CNN architectures."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseCNN(nn.Module, ABC):
    """Abstract base class for CNN models.

    All CNN models in this package should inherit from this class and implement
    the required methods.
    """

    def __init__(self) -> None:
        """Initialize the base CNN model."""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        pass

    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

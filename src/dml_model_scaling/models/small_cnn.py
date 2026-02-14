"""Small CNN architecture with approximately 10K parameters."""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Small convolutional neural network for MNIST classification.

    Architecture:
        - Conv layer 1: 1 -> 8 channels, 3x3 kernel
        - MaxPool: 2x2
        - Conv layer 2: 8 -> 16 channels, 3x3 kernel
        - MaxPool: 2x2
        - Fully connected: 784 -> 10

    Total parameters: ~10,634
    """

    def __init__(self) -> None:
        """Initialize the SmallCNN model."""
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Output logits of shape (batch_size, 10).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Count the number of trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

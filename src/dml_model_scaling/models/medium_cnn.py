"""Medium CNN architecture with approximately 40K parameters."""

import torch
import torch.nn as nn


class MediumCNN(nn.Module):
    """Medium convolutional neural network for MNIST classification.

    Architecture:
        - Conv layer 1: 1 -> 16 channels, 3x3 kernel
        - MaxPool: 2x2
        - Conv layer 2: 16 -> 32 channels, 3x3 kernel
        - MaxPool: 2x2
        - Conv layer 3: 32 -> 64 channels, 3x3 kernel
        - Fully connected: 3136 -> 128 -> 10

    Total parameters: ~39,530
    """

    def __init__(self) -> None:
        """Initialize the MediumCNN model."""
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
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

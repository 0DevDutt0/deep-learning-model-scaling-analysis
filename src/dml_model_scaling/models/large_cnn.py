"""Large CNN architecture with approximately 160K parameters."""

import torch
import torch.nn as nn


class LargeCNN(nn.Module):
    """Large convolutional neural network for MNIST classification.

    Architecture:
        - Conv layer 1: 1 -> 32 channels, 3x3 kernel
        - MaxPool: 2x2
        - Conv layer 2: 32 -> 64 channels, 3x3 kernel
        - MaxPool: 2x2
        - Conv layer 3: 64 -> 128 channels, 3x3 kernel
        - Conv layer 4: 128 -> 128 channels, 3x3 kernel
        - Fully connected: 6272 -> 256 -> 10

    Total parameters: ~166,410
    """

    def __init__(self) -> None:
        """Initialize the LargeCNN model."""
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
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

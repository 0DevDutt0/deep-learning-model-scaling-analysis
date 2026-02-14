"""Utilities for managing random seeds and reproducibility."""

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.

    Examples:
        >>> set_random_seed(42)
        >>> # All subsequent random operations will be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

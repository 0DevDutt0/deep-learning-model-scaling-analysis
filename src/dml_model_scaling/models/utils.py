"""Utility functions for working with models."""

import torch.nn as nn

from dml_model_scaling.exceptions import ModelError


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model.

    Args:
        model: The PyTorch model to analyze.

    Returns:
        The total number of trainable parameters.

    Examples:
        >>> from dml_model_scaling.models import SmallCNN
        >>> model = SmallCNN()
        >>> count_parameters(model)
        10634
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_by_name(name: str) -> nn.Module:
    """Get a model instance by name.

    Args:
        name: Name of the model. One of 'small', 'medium', 'large'.

    Returns:
        An instance of the requested model.

    Raises:
        ModelError: If the model name is not recognized.

    Examples:
        >>> model = get_model_by_name('small')
        >>> isinstance(model, nn.Module)
        True
    """
    from dml_model_scaling.models import MODEL_REGISTRY

    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        valid_names = ", ".join(MODEL_REGISTRY.keys())
        raise ModelError(
            f"Unknown model name: '{name}'. Valid options are: {valid_names}"
        )

    return MODEL_REGISTRY[name_lower]()

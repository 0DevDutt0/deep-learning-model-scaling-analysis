"""Neural network models for scaling experiments."""

from typing import Union

import torch.nn as nn

from dml_model_scaling.models.large_cnn import LargeCNN
from dml_model_scaling.models.medium_cnn import MediumCNN
from dml_model_scaling.models.small_cnn import SmallCNN
from dml_model_scaling.models.utils import count_parameters, get_model_by_name

__all__ = [
    "SmallCNN",
    "MediumCNN",
    "LargeCNN",
    "count_parameters",
    "get_model_by_name",
    "MODEL_REGISTRY",
]

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "small": SmallCNN,
    "medium": MediumCNN,
    "large": LargeCNN,
}

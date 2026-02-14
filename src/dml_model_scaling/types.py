"""Common type aliases used throughout the package."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

DeviceType = Union[str, torch.device]
PathLike = Union[str, Path]
TensorArray = Union[torch.Tensor, NDArray[np.float32]]

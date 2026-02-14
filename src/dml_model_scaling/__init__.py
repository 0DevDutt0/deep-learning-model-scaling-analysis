"""Deep Learning Model Scaling Analysis: Causal inference analysis of neural network scaling laws.

This package provides tools for conducting controlled experiments on neural network
architectures and applying Double Machine Learning to estimate causal effects of
model size on performance.
"""

__version__ = "0.1.0"
__author__ = "Deep Learning Model Scaling Analysis Contributors"
__all__ = [
    "SmallCNN",
    "MediumCNN",
    "LargeCNN",
    "ExperimentRunner",
    "DMLAnalyzer",
    "__version__",
]

from dml_model_scaling.models import LargeCNN, MediumCNN, SmallCNN

try:
    from dml_model_scaling.analysis.dml import DMLAnalyzer
    from dml_model_scaling.experiments.runner import ExperimentRunner
except ImportError:
    pass

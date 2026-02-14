"""Configuration models for experiments."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment or experiment suite.

    Attributes:
        model_names: List of model names to train (small, medium, large).
        dataset_sizes: List of dataset sizes to use for training.
        epochs_list: List of epoch counts to try.
        learning_rates: List of learning rates to try.
        batch_size: Batch size for training.
        device: Device to use for training (cpu, cuda, mps, auto).
        random_seed: Random seed for reproducibility.
        output_path: Path to save experiment results CSV.
        data_dir: Directory containing or for downloading MNIST data.
    """

    model_names: list[Literal["small", "medium", "large"]] = Field(
        default=["small", "medium", "large"],
        description="Models to train",
    )

    dataset_sizes: list[int] = Field(
        default=[2000, 5000, 8000],
        description="Dataset sizes to experiment with",
    )

    epochs_list: list[int] = Field(
        default=[3, 5],
        description="Number of epochs for each experiment",
    )

    learning_rates: list[float] = Field(
        default=[0.001, 0.0005],
        description="Learning rates to try",
    )

    batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size for training",
    )

    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Device for training",
    )

    random_seed: int = Field(
        default=42,
        description="Random seed",
    )

    output_path: Path = Field(
        default=Path("data/experiments.csv"),
        description="Path to save results",
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Data directory",
    )

    @field_validator("dataset_sizes", "epochs_list")
    @classmethod
    def validate_positive_list(cls, v: list[int]) -> list[int]:
        """Validate that all values in the list are positive."""
        if any(x <= 0 for x in v):
            raise ValueError("All values must be positive")
        return v

    @field_validator("learning_rates")
    @classmethod
    def validate_learning_rates(cls, v: list[float]) -> list[float]:
        """Validate that learning rates are positive and reasonable."""
        if any(lr <= 0 or lr > 1 for lr in v):
            raise ValueError("Learning rates must be in range (0, 1]")
        return v

    def total_experiments(self) -> int:
        """Calculate the total number of experiments to run.

        Returns:
            Total number of experiments.
        """
        return (
            len(self.model_names)
            * len(self.dataset_sizes)
            * len(self.epochs_list)
            * len(self.learning_rates)
        )


class AnalysisConfig(BaseModel):
    """Configuration for DML causal analysis.

    Attributes:
        input_path: Path to experiments CSV file.
        n_estimators: Number of trees in random forest models.
        max_depth: Maximum depth of random forest trees.
        cv_folds: Number of cross-validation folds for DML.
        random_seed: Random seed for reproducibility.
    """

    input_path: Path = Field(
        default=Path("data/experiments.csv"),
        description="Path to experiments CSV",
    )

    n_estimators: int = Field(
        default=200,
        ge=10,
        description="Number of trees in random forests",
    )

    max_depth: int = Field(
        default=5,
        ge=1,
        description="Maximum depth of trees",
    )

    cv_folds: int = Field(
        default=3,
        ge=2,
        description="Number of CV folds for DML",
    )

    random_seed: int = Field(
        default=42,
        description="Random seed",
    )

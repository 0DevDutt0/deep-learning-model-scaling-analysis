"""Experiment results management and storage."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from dml_model_scaling.exceptions import DataError
from dml_model_scaling.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single experiment.

    Attributes:
        experiment_id: Unique identifier for the experiment.
        model_name: Name of the model (e.g., 'small', 'medium', 'large').
        model_size: Number of parameters in the model.
        dataset_size: Number of training samples used.
        epochs: Number of training epochs.
        learning_rate: Learning rate used.
        batch_size: Batch size used.
        training_time: Time taken to train in seconds.
        accuracy: Final test accuracy.
    """

    experiment_id: int
    model_name: str
    model_size: int
    dataset_size: int
    epochs: int
    learning_rate: float
    batch_size: int
    training_time: float
    accuracy: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "dataset_size": self.dataset_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "training_time": round(self.training_time, 2),
            "accuracy": round(self.accuracy, 4),
        }


class ResultsManager:
    """Manages storage and retrieval of experiment results."""

    def __init__(self, output_path: Path) -> None:
        """Initialize the results manager.

        Args:
            output_path: Path to the CSV file for storing results.
        """
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def initialize_csv(self) -> None:
        """Initialize the CSV file with headers."""
        if self._initialized:
            return

        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "experiment_id",
                "model_name",
                "model_size",
                "dataset_size",
                "epochs",
                "learning_rate",
                "batch_size",
                "training_time",
                "accuracy",
            ])

        self._initialized = True
        logger.info(f"Initialized results CSV at {self.output_path}")

    def save_result(self, result: ExperimentResult) -> None:
        """Save a single experiment result to CSV.

        Args:
            result: The experiment result to save.
        """
        if not self._initialized:
            self.initialize_csv()

        with open(self.output_path, "a", newline="") as f:
            writer = csv.writer(f)
            result_dict = result.to_dict()
            writer.writerow([
                result_dict["experiment_id"],
                result_dict["model_name"],
                result_dict["model_size"],
                result_dict["dataset_size"],
                result_dict["epochs"],
                result_dict["learning_rate"],
                result_dict["batch_size"],
                result_dict["training_time"],
                result_dict["accuracy"],
            ])

    def load_results(self) -> pd.DataFrame:
        """Load all results from CSV.

        Returns:
            DataFrame containing all experiment results.

        Raises:
            DataError: If the CSV file doesn't exist or can't be read.
        """
        if not self.output_path.exists():
            raise DataError(f"Results file not found: {self.output_path}")

        try:
            df = pd.read_csv(self.output_path)
            logger.info(f"Loaded {len(df)} experiment results from {self.output_path}")
            return df
        except Exception as e:
            raise DataError(f"Failed to load results: {e}") from e

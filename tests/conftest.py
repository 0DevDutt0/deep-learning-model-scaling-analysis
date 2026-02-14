"""Shared test fixtures and configuration."""

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dml_model_scaling.config.experiment_config import AnalysisConfig, ExperimentConfig
from dml_model_scaling.models import LargeCNN, MediumCNN, SmallCNN


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_path(temp_dir: Path) -> Path:
    """Path for a sample CSV file."""
    return temp_dir / "test_experiments.csv"


@pytest.fixture
def sample_experiment_data() -> pd.DataFrame:
    """Create sample experiment data for testing."""
    data = {
        "experiment_id": [1, 2, 3, 4, 5, 6],
        "model_name": ["small", "small", "medium", "medium", "large", "large"],
        "model_size": [10634, 10634, 39530, 39530, 166410, 166410],
        "dataset_size": [1000, 2000, 1000, 2000, 1000, 2000],
        "epochs": [3, 3, 5, 5, 3, 3],
        "learning_rate": [0.001, 0.001, 0.0005, 0.0005, 0.001, 0.001],
        "batch_size": [64, 64, 64, 64, 64, 64],
        "training_time": [10.5, 20.3, 15.2, 30.1, 25.5, 50.2],
        "accuracy": [0.85, 0.87, 0.90, 0.92, 0.93, 0.95],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(sample_csv_path: Path, sample_experiment_data: pd.DataFrame) -> Path:
    """Create a sample CSV file with experiment data."""
    sample_experiment_data.to_csv(sample_csv_path, index=False)
    return sample_csv_path


@pytest.fixture
def small_cnn() -> SmallCNN:
    """Create a SmallCNN model instance."""
    return SmallCNN()


@pytest.fixture
def medium_cnn() -> MediumCNN:
    """Create a MediumCNN model instance."""
    return MediumCNN()


@pytest.fixture
def large_cnn() -> LargeCNN:
    """Create a LargeCNN model instance."""
    return LargeCNN()


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a sample batch of MNIST-like data."""
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    return x, y


@pytest.fixture
def sample_dataloader() -> DataLoader:
    """Create a sample DataLoader with dummy data."""
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def experiment_config(temp_dir: Path) -> ExperimentConfig:
    """Create a test experiment configuration."""
    return ExperimentConfig(
        model_names=["small"],
        dataset_sizes=[100],
        epochs_list=[1],
        learning_rates=[0.001],
        batch_size=16,
        device="cpu",
        random_seed=42,
        output_path=temp_dir / "test_results.csv",
        data_dir=temp_dir / "data",
    )


@pytest.fixture
def analysis_config(sample_csv_path: Path) -> AnalysisConfig:
    """Create a test analysis configuration."""
    return AnalysisConfig(
        input_path=sample_csv_path,
        n_estimators=10,
        max_depth=3,
        cv_folds=2,
        random_seed=42,
    )

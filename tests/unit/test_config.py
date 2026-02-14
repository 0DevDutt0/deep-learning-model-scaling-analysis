"""Unit tests for configuration modules."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dml_model_scaling.config.experiment_config import AnalysisConfig, ExperimentConfig


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_default_config(self) -> None:
        """Test creating config with default values."""
        config = ExperimentConfig()
        assert config.model_names == ["small", "medium", "large"]
        assert config.batch_size == 64
        assert config.random_seed == 42

    def test_custom_config(self) -> None:
        """Test creating config with custom values."""
        config = ExperimentConfig(
            model_names=["small"],
            dataset_sizes=[1000],
            epochs_list=[2],
            learning_rates=[0.001],
            batch_size=32,
        )
        assert config.model_names == ["small"]
        assert config.dataset_sizes == [1000]
        assert config.batch_size == 32

    def test_total_experiments(self) -> None:
        """Test calculation of total experiments."""
        config = ExperimentConfig(
            model_names=["small", "medium"],
            dataset_sizes=[1000, 2000],
            epochs_list=[3],
            learning_rates=[0.001, 0.0005],
        )
        assert config.total_experiments() == 2 * 2 * 1 * 2  # 8

    def test_invalid_dataset_sizes(self) -> None:
        """Test validation of dataset sizes."""
        with pytest.raises(ValidationError):
            ExperimentConfig(dataset_sizes=[0, 1000])

    def test_invalid_epochs(self) -> None:
        """Test validation of epochs."""
        with pytest.raises(ValidationError):
            ExperimentConfig(epochs_list=[-1, 5])

    def test_invalid_learning_rates(self) -> None:
        """Test validation of learning rates."""
        with pytest.raises(ValidationError):
            ExperimentConfig(learning_rates=[0.0, 0.001])

        with pytest.raises(ValidationError):
            ExperimentConfig(learning_rates=[2.0])

    def test_invalid_batch_size(self) -> None:
        """Test validation of batch size."""
        with pytest.raises(ValidationError):
            ExperimentConfig(batch_size=0)


class TestAnalysisConfig:
    """Tests for AnalysisConfig."""

    def test_default_config(self) -> None:
        """Test creating config with default values."""
        config = AnalysisConfig()
        assert config.n_estimators == 200
        assert config.max_depth == 5
        assert config.cv_folds == 3
        assert config.random_seed == 42

    def test_custom_config(self) -> None:
        """Test creating config with custom values."""
        config = AnalysisConfig(
            input_path=Path("custom.csv"),
            n_estimators=100,
            max_depth=3,
            cv_folds=5,
        )
        assert config.input_path == Path("custom.csv")
        assert config.n_estimators == 100
        assert config.cv_folds == 5

    def test_invalid_n_estimators(self) -> None:
        """Test validation of n_estimators."""
        with pytest.raises(ValidationError):
            AnalysisConfig(n_estimators=5)

    def test_invalid_max_depth(self) -> None:
        """Test validation of max_depth."""
        with pytest.raises(ValidationError):
            AnalysisConfig(max_depth=0)

    def test_invalid_cv_folds(self) -> None:
        """Test validation of cv_folds."""
        with pytest.raises(ValidationError):
            AnalysisConfig(cv_folds=1)

"""Unit tests for experiment modules."""

import pytest
import torch

from dml_model_scaling.experiments.results import ExperimentResult, ResultsManager
from dml_model_scaling.experiments.trainer import ModelTrainer, get_device
from dml_model_scaling.exceptions import DataError, ModelError


class TestGetDevice:
    """Tests for device selection."""

    def test_get_device_cpu(self) -> None:
        """Test getting CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_auto(self) -> None:
        """Test automatic device selection."""
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]


class TestModelTrainer:
    """Tests for ModelTrainer."""

    def test_initialization(self, small_cnn: torch.nn.Module) -> None:
        """Test trainer initialization."""
        trainer = ModelTrainer(small_cnn, learning_rate=0.001, device="cpu")
        assert trainer.device.type == "cpu"
        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_train_epoch(
        self,
        small_cnn: torch.nn.Module,
        sample_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Test training for one epoch."""
        trainer = ModelTrainer(small_cnn, learning_rate=0.001, device="cpu")
        loss = trainer.train_epoch(sample_dataloader)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate(
        self,
        small_cnn: torch.nn.Module,
        sample_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Test model evaluation."""
        trainer = ModelTrainer(small_cnn, learning_rate=0.001, device="cpu")
        accuracy, loss = trainer.evaluate(sample_dataloader)
        assert isinstance(accuracy, float)
        assert isinstance(loss, float)
        assert 0 <= accuracy <= 1
        assert loss >= 0

    def test_train_invalid_epochs(
        self,
        small_cnn: torch.nn.Module,
        sample_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Test that training with invalid epochs raises error."""
        trainer = ModelTrainer(small_cnn, learning_rate=0.001, device="cpu")
        with pytest.raises(ModelError, match="must be positive"):
            trainer.train(sample_dataloader, sample_dataloader, epochs=0)


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_creation(self) -> None:
        """Test creating an ExperimentResult."""
        result = ExperimentResult(
            experiment_id=1,
            model_name="small",
            model_size=10000,
            dataset_size=1000,
            epochs=5,
            learning_rate=0.001,
            batch_size=64,
            training_time=10.5,
            accuracy=0.85,
        )
        assert result.experiment_id == 1
        assert result.model_name == "small"
        assert result.accuracy == 0.85

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = ExperimentResult(
            experiment_id=1,
            model_name="small",
            model_size=10000,
            dataset_size=1000,
            epochs=5,
            learning_rate=0.001,
            batch_size=64,
            training_time=10.5,
            accuracy=0.85,
        )
        result_dict = result.to_dict()
        assert result_dict["experiment_id"] == 1
        assert result_dict["model_name"] == "small"
        assert result_dict["training_time"] == 10.5
        assert result_dict["accuracy"] == 0.85


class TestResultsManager:
    """Tests for ResultsManager."""

    def test_initialization(self, sample_csv_path: pytest.fixture) -> None:
        """Test ResultsManager initialization."""
        manager = ResultsManager(sample_csv_path)
        assert manager.output_path == sample_csv_path

    def test_initialize_csv(self, sample_csv_path: pytest.fixture) -> None:
        """Test CSV initialization."""
        manager = ResultsManager(sample_csv_path)
        manager.initialize_csv()
        assert sample_csv_path.exists()

    def test_save_result(self, sample_csv_path: pytest.fixture) -> None:
        """Test saving a result."""
        manager = ResultsManager(sample_csv_path)
        result = ExperimentResult(
            experiment_id=1,
            model_name="small",
            model_size=10000,
            dataset_size=1000,
            epochs=5,
            learning_rate=0.001,
            batch_size=64,
            training_time=10.5,
            accuracy=0.85,
        )
        manager.save_result(result)
        assert sample_csv_path.exists()

    def test_load_results_nonexistent(self, sample_csv_path: pytest.fixture) -> None:
        """Test loading from nonexistent file raises error."""
        manager = ResultsManager(sample_csv_path)
        with pytest.raises(DataError, match="not found"):
            manager.load_results()

    def test_load_results(self, sample_csv_file: pytest.fixture) -> None:
        """Test loading results from CSV."""
        manager = ResultsManager(sample_csv_file)
        df = manager.load_results()
        assert len(df) == 6
        assert "accuracy" in df.columns

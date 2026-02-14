"""Unit tests for CNN models."""

import pytest
import torch

from dml_model_scaling.models import LargeCNN, MediumCNN, SmallCNN
from dml_model_scaling.models.utils import count_parameters, get_model_by_name
from dml_model_scaling.exceptions import ModelError


class TestSmallCNN:
    """Tests for SmallCNN model."""

    def test_initialization(self, small_cnn: SmallCNN) -> None:
        """Test that SmallCNN initializes correctly."""
        assert isinstance(small_cnn, torch.nn.Module)
        assert hasattr(small_cnn, "features")
        assert hasattr(small_cnn, "classifier")

    def test_parameter_count(self, small_cnn: SmallCNN) -> None:
        """Test that SmallCNN has approximately 10K parameters."""
        param_count = count_parameters(small_cnn)
        assert 9_000 < param_count < 12_000, f"Expected ~10K params, got {param_count}"

    def test_forward_pass_shape(self, small_cnn: SmallCNN, sample_batch: tuple) -> None:
        """Test forward pass produces correct output shape."""
        x, _ = sample_batch
        output = small_cnn(x)
        assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"

    def test_forward_pass_values(self, small_cnn: SmallCNN, sample_batch: tuple) -> None:
        """Test forward pass produces valid logits."""
        x, _ = sample_batch
        output = small_cnn(x)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"


class TestMediumCNN:
    """Tests for MediumCNN model."""

    def test_initialization(self, medium_cnn: MediumCNN) -> None:
        """Test that MediumCNN initializes correctly."""
        assert isinstance(medium_cnn, torch.nn.Module)
        assert hasattr(medium_cnn, "features")
        assert hasattr(medium_cnn, "classifier")

    def test_parameter_count(self, medium_cnn: MediumCNN) -> None:
        """Test that MediumCNN has approximately 400K parameters."""
        param_count = count_parameters(medium_cnn)
        assert 400_000 < param_count < 450_000, f"Expected ~400K params, got {param_count}"

    def test_forward_pass_shape(self, medium_cnn: MediumCNN, sample_batch: tuple) -> None:
        """Test forward pass produces correct output shape."""
        x, _ = sample_batch
        output = medium_cnn(x)
        assert output.shape == (4, 10)


class TestLargeCNN:
    """Tests for LargeCNN model."""

    def test_initialization(self, large_cnn: LargeCNN) -> None:
        """Test that LargeCNN initializes correctly."""
        assert isinstance(large_cnn, torch.nn.Module)
        assert hasattr(large_cnn, "features")
        assert hasattr(large_cnn, "classifier")

    def test_parameter_count(self, large_cnn: LargeCNN) -> None:
        """Test that LargeCNN has approximately 1.8M parameters."""
        param_count = count_parameters(large_cnn)
        assert 1_800_000 < param_count < 1_900_000, f"Expected ~1.8M params, got {param_count}"

    def test_forward_pass_shape(self, large_cnn: LargeCNN, sample_batch: tuple) -> None:
        """Test forward pass produces correct output shape."""
        x, _ = sample_batch
        output = large_cnn(x)
        assert output.shape == (4, 10)


class TestModelUtils:
    """Tests for model utility functions."""

    def test_count_parameters(self, small_cnn: SmallCNN) -> None:
        """Test parameter counting function."""
        count = count_parameters(small_cnn)
        assert isinstance(count, int)
        assert count > 0

    def test_get_model_by_name_small(self) -> None:
        """Test getting SmallCNN by name."""
        model = get_model_by_name("small")
        assert isinstance(model, SmallCNN)

    def test_get_model_by_name_medium(self) -> None:
        """Test getting MediumCNN by name."""
        model = get_model_by_name("medium")
        assert isinstance(model, MediumCNN)

    def test_get_model_by_name_large(self) -> None:
        """Test getting LargeCNN by name."""
        model = get_model_by_name("large")
        assert isinstance(model, LargeCNN)

    def test_get_model_by_name_case_insensitive(self) -> None:
        """Test that model name lookup is case-insensitive."""
        model1 = get_model_by_name("Small")
        model2 = get_model_by_name("SMALL")
        assert isinstance(model1, SmallCNN)
        assert isinstance(model2, SmallCNN)

    def test_get_model_by_name_invalid(self) -> None:
        """Test that invalid model name raises error."""
        with pytest.raises(ModelError, match="Unknown model name"):
            get_model_by_name("invalid_model")

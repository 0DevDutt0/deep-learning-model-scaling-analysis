"""Unit tests for analysis modules."""

import pytest

from dml_model_scaling.analysis.dml import DMLAnalyzer
from dml_model_scaling.config.experiment_config import AnalysisConfig
from dml_model_scaling.exceptions import AnalysisError, DataError


class TestDMLAnalyzer:
    """Tests for DMLAnalyzer."""

    def test_initialization(self, analysis_config: AnalysisConfig) -> None:
        """Test analyzer initialization."""
        analyzer = DMLAnalyzer(analysis_config)
        assert analyzer.config == analysis_config
        assert analyzer.data is None
        assert analyzer.dml_model is None

    def test_load_data_nonexistent(self, analysis_config: AnalysisConfig) -> None:
        """Test loading nonexistent file raises error."""
        analyzer = DMLAnalyzer(analysis_config)
        with pytest.raises(DataError, match="not found"):
            analyzer.load_data()

    def test_load_data(
        self,
        sample_csv_file: pytest.fixture,
        analysis_config: AnalysisConfig,
    ) -> None:
        """Test loading data from CSV."""
        analyzer = DMLAnalyzer(analysis_config)
        df = analyzer.load_data(sample_csv_file)
        assert len(df) == 6
        assert "accuracy" in df.columns
        assert "model_size" in df.columns

    def test_fit_without_data(self, analysis_config: AnalysisConfig) -> None:
        """Test that fitting without loading data raises error."""
        analyzer = DMLAnalyzer(analysis_config)
        with pytest.raises(AnalysisError, match="No data loaded"):
            analyzer.fit()

    @pytest.mark.slow
    def test_analyze(
        self,
        sample_csv_file: pytest.fixture,
        analysis_config: AnalysisConfig,
    ) -> None:
        """Test full analysis pipeline."""
        analyzer = DMLAnalyzer(analysis_config)
        results = analyzer.analyze(sample_csv_file)

        assert results is not None
        assert isinstance(results.average_effect, float)
        assert isinstance(results.effect_per_million, float)
        assert results.num_samples == 6
        assert results.cv_folds == 2
        assert len(results.confounders_used) == 5

    def test_dml_results_str(
        self,
        sample_csv_file: pytest.fixture,
        analysis_config: AnalysisConfig,
    ) -> None:
        """Test string representation of results."""
        analyzer = DMLAnalyzer(analysis_config)
        results = analyzer.analyze(sample_csv_file)
        result_str = str(results)

        assert "DML Causal Analysis Results" in result_str
        assert "Average causal effect" in result_str
        assert "Effect per +1M parameters" in result_str

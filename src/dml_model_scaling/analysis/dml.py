"""Double Machine Learning causal inference analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from dml_model_scaling.config.experiment_config import AnalysisConfig
from dml_model_scaling.exceptions import AnalysisError, DataError
from dml_model_scaling.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DMLResults:
    """Results from DML causal analysis.

    Attributes:
        average_effect: Average causal effect of model size on accuracy.
        effect_per_million: Effect scaled to per-million-parameter change.
        confounders_used: List of confounder variable names.
        num_samples: Number of samples used in analysis.
        cv_folds: Number of cross-validation folds used.
    """

    average_effect: float
    effect_per_million: float
    confounders_used: list[str]
    num_samples: int
    cv_folds: int

    def __str__(self) -> str:
        """Format results as a readable string."""
        return (
            f"DML Causal Analysis Results\n"
            f"{'=' * 50}\n"
            f"Average causal effect: {self.average_effect:.8f}\n"
            f"Effect per +1M parameters: {self.effect_per_million:.4f}\n"
            f"Samples analyzed: {self.num_samples}\n"
            f"CV folds: {self.cv_folds}\n"
            f"Confounders: {', '.join(self.confounders_used)}\n"
        )


class DMLAnalyzer:
    """Performs Double Machine Learning causal analysis.

    This class applies the DML methodology to estimate the causal effect of
    model size (number of parameters) on test accuracy, while controlling for
    confounding variables like training time, dataset size, etc.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the DML analyzer.

        Args:
            config: Configuration for the analysis.
        """
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.dml_model: Optional[LinearDML] = None

    def load_data(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load experiment results from CSV.

        Args:
            path: Path to CSV file. If None, uses config.input_path.

        Returns:
            DataFrame containing experiment results.

        Raises:
            DataError: If data loading fails.
        """
        csv_path = path or self.config.input_path

        if not csv_path.exists():
            raise DataError(f"Input file not found: {csv_path}")

        try:
            self.data = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.data)} experiment results from {csv_path}")

            required_columns = [
                "accuracy",
                "model_size",
                "training_time",
                "dataset_size",
                "epochs",
                "batch_size",
                "learning_rate",
            ]

            missing_columns = set(required_columns) - set(self.data.columns)
            if missing_columns:
                raise DataError(f"Missing required columns: {missing_columns}")

            return self.data

        except Exception as e:
            raise DataError(f"Failed to load data: {e}") from e

    def _prepare_variables(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Prepare Y, T, and X variables for DML.

        Returns:
            Tuple of (Y, T, X) arrays.

        Raises:
            AnalysisError: If data preparation fails.
        """
        if self.data is None:
            raise AnalysisError("No data loaded. Call load_data() first.")

        try:
            Y = self.data["accuracy"].values.ravel()

            T = self.data["model_size"].values.reshape(-1, 1)

            confounders = [
                "training_time",
                "dataset_size",
                "epochs",
                "batch_size",
                "learning_rate",
            ]
            X = self.data[confounders].values

            logger.info(
                f"Prepared variables: Y shape={Y.shape}, T shape={T.shape}, X shape={X.shape}"
            )

            return Y, T, X

        except Exception as e:
            raise AnalysisError(f"Failed to prepare variables: {e}") from e

    def fit(self) -> DMLResults:
        """Fit the DML model and estimate causal effects.

        Returns:
            DML analysis results.

        Raises:
            AnalysisError: If model fitting fails.
        """
        if self.data is None:
            raise AnalysisError("No data loaded. Call load_data() first.")

        try:
            Y, T, X = self._prepare_variables()

            model_y = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_seed,
            )

            model_t = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_seed,
            )

            self.dml_model = LinearDML(
                model_y=model_y,
                model_t=model_t,
                random_state=self.config.random_seed,
                cv=self.config.cv_folds,
            )

            logger.info("Fitting DML model...")
            self.dml_model.fit(Y, T, X=X)

            effect = self.dml_model.const_marginal_effect(X)
            average_effect = float(effect.mean())
            effect_per_million = average_effect * 1_000_000

            confounders = [
                "training_time",
                "dataset_size",
                "epochs",
                "batch_size",
                "learning_rate",
            ]

            results = DMLResults(
                average_effect=average_effect,
                effect_per_million=effect_per_million,
                confounders_used=confounders,
                num_samples=len(self.data),
                cv_folds=self.config.cv_folds,
            )

            logger.info("DML analysis completed successfully")
            return results

        except Exception as e:
            raise AnalysisError(f"Failed to fit DML model: {e}") from e

    def analyze(self, path: Optional[Path] = None) -> DMLResults:
        """Load data, fit model, and return results in one call.

        Args:
            path: Path to CSV file. If None, uses config.input_path.

        Returns:
            DML analysis results.
        """
        self.load_data(path)
        return self.fit()

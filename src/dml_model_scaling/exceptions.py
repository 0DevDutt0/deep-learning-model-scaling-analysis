"""Custom exceptions for the DML Model Scaling package."""


class DMLModelScalingError(Exception):
    """Base exception for all DML Model Scaling errors."""

    pass


class ConfigurationError(DMLModelScalingError):
    """Raised when configuration is invalid or incomplete."""

    pass


class ModelError(DMLModelScalingError):
    """Raised when there's an error with model creation or training."""

    pass


class DataError(DMLModelScalingError):
    """Raised when there's an error with data loading or processing."""

    pass


class AnalysisError(DMLModelScalingError):
    """Raised when there's an error during causal analysis."""

    pass


class ValidationError(DMLModelScalingError):
    """Raised when validation fails."""

    pass

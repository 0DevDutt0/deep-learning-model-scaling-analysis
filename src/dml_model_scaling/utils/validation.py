"""Input validation utilities."""

from pathlib import Path
from typing import Any

from dml_model_scaling.exceptions import ValidationError


def validate_positive_int(value: Any, name: str) -> int:
    """Validate that a value is a positive integer.

    Args:
        value: Value to validate.
        name: Name of the parameter for error messages.

    Returns:
        The validated integer value.

    Raises:
        ValidationError: If validation fails.
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}") from e

    if int_value <= 0:
        raise ValidationError(f"{name} must be positive, got {int_value}")

    return int_value


def validate_positive_float(value: Any, name: str) -> float:
    """Validate that a value is a positive float.

    Args:
        value: Value to validate.
        name: Name of the parameter for error messages.

    Returns:
        The validated float value.

    Raises:
        ValidationError: If validation fails.
    """
    try:
        float_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}") from e

    if float_value <= 0:
        raise ValidationError(f"{name} must be positive, got {float_value}")

    return float_value


def validate_file_exists(path: Path, name: str) -> Path:
    """Validate that a file exists.

    Args:
        path: Path to validate.
        name: Name of the parameter for error messages.

    Returns:
        The validated Path object.

    Raises:
        ValidationError: If the file doesn't exist.
    """
    if not path.exists():
        raise ValidationError(f"{name} does not exist: {path}")

    if not path.is_file():
        raise ValidationError(f"{name} is not a file: {path}")

    return path


def validate_in_range(
    value: float,
    name: str,
    min_value: float,
    max_value: float,
) -> float:
    """Validate that a value is within a specified range.

    Args:
        value: Value to validate.
        name: Name of the parameter for error messages.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).

    Returns:
        The validated value.

    Raises:
        ValidationError: If the value is out of range.
    """
    if not min_value <= value <= max_value:
        raise ValidationError(
            f"{name} must be in range [{min_value}, {max_value}], got {value}"
        )

    return value

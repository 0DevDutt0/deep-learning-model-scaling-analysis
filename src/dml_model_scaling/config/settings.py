"""Application settings using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings.

    Settings can be configured via environment variables with the DML_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="DML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Root directory of the project",
    )

    data_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "data",
        description="Directory for storing data",
    )

    output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "outputs",
        description="Directory for storing outputs",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Device to use for training",
    )

    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    num_workers: int = Field(
        default=4,
        description="Number of workers for data loading",
    )

    def __init__(self, **kwargs: object) -> None:
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        The application settings.
    """
    return Settings()

"""Training CLI commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from dml_model_scaling.config.experiment_config import ExperimentConfig
from dml_model_scaling.experiments.runner import ExperimentRunner
from dml_model_scaling.utils.logging import setup_logging

app = typer.Typer()
console = Console()


@app.command()
def run(
    output: Path = typer.Option(
        Path("data/experiments.csv"),
        "--output",
        "-o",
        help="Output CSV file path",
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models",
        "-m",
        help="Comma-separated list of models (small,medium,large)",
    ),
    dataset_sizes: Optional[str] = typer.Option(
        None,
        "--dataset-sizes",
        "-d",
        help="Comma-separated list of dataset sizes",
    ),
    epochs: Optional[str] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Comma-separated list of epoch counts",
    ),
    learning_rates: Optional[str] = typer.Option(
        None,
        "--learning-rates",
        "-lr",
        help="Comma-separated list of learning rates",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        "-b",
        help="Batch size for training",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device for training (cpu, cuda, mps, auto)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
    ),
) -> None:
    """Run training experiments with specified configuration.

    Examples:
        # Run with default configuration
        dml-scale train run

        # Run with custom models and dataset sizes
        dml-scale train run --models small,medium --dataset-sizes 1000,2000

        # Run with custom output path
        dml-scale train run --output results/my_experiments.csv
    """
    setup_logging(level=log_level)

    config_dict = {
        "output_path": output,
        "batch_size": batch_size,
        "device": device,
    }

    if models:
        config_dict["model_names"] = models.split(",")

    if dataset_sizes:
        config_dict["dataset_sizes"] = [int(x) for x in dataset_sizes.split(",")]

    if epochs:
        config_dict["epochs_list"] = [int(x) for x in epochs.split(",")]

    if learning_rates:
        config_dict["learning_rates"] = [float(x) for x in learning_rates.split(",")]

    try:
        config = ExperimentConfig(**config_dict)

        console.print("\n[bold cyan]Experiment Configuration[/bold cyan]")
        console.print(f"Models: {config.model_names}")
        console.print(f"Dataset sizes: {config.dataset_sizes}")
        console.print(f"Epochs: {config.epochs_list}")
        console.print(f"Learning rates: {config.learning_rates}")
        console.print(f"Batch size: {config.batch_size}")
        console.print(f"Device: {config.device}")
        console.print(f"Total experiments: {config.total_experiments()}\n")

        runner = ExperimentRunner(config)
        output_path = runner.run()

        console.print(f"[bold green]✓ Success![/bold green] Results: {output_path}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

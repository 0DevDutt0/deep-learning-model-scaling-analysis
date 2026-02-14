"""Analysis CLI commands."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dml_model_scaling.analysis.dml import DMLAnalyzer
from dml_model_scaling.config.experiment_config import AnalysisConfig
from dml_model_scaling.utils.logging import setup_logging

app = typer.Typer()
console = Console()


@app.command()
def run(
    input_file: Path = typer.Option(
        Path("data/experiments.csv"),
        "--input",
        "-i",
        help="Input CSV file with experiment results",
    ),
    n_estimators: int = typer.Option(
        200,
        "--n-estimators",
        help="Number of trees in random forests",
    ),
    max_depth: int = typer.Option(
        5,
        "--max-depth",
        help="Maximum depth of trees",
    ),
    cv_folds: int = typer.Option(
        3,
        "--cv-folds",
        help="Number of cross-validation folds",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
    ),
) -> None:
    """Run DML causal analysis on experiment results.

    Examples:
        # Analyze default experiments file
        dml-scale analyze run

        # Analyze custom file with more estimators
        dml-scale analyze run --input my_results.csv --n-estimators 500

        # Use more CV folds
        dml-scale analyze run --cv-folds 5
    """
    setup_logging(level=log_level)

    if not input_file.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {input_file}")
        raise typer.Exit(1)

    try:
        config = AnalysisConfig(
            input_path=input_file,
            n_estimators=n_estimators,
            max_depth=max_depth,
            cv_folds=cv_folds,
        )

        console.print("\n[bold cyan]DML Analysis Configuration[/bold cyan]")
        console.print(f"Input: {input_file}")
        console.print(f"Random Forest estimators: {n_estimators}")
        console.print(f"Max depth: {max_depth}")
        console.print(f"CV folds: {cv_folds}\n")

        analyzer = DMLAnalyzer(config)
        results = analyzer.analyze()

        table = Table(title="DML Causal Analysis Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Average Causal Effect", f"{results.average_effect:.8f}")
        table.add_row("Effect per +1M Parameters", f"{results.effect_per_million:.4f}")
        table.add_row("Samples Analyzed", str(results.num_samples))
        table.add_row("CV Folds", str(results.cv_folds))
        table.add_row("Confounders", ", ".join(results.confounders_used))

        console.print(table)
        console.print()

        interpretation = (
            f"[bold]Interpretation:[/bold] Adding 1 million parameters to the model "
            f"is estimated to improve accuracy by approximately "
            f"[bold green]{results.effect_per_million:.4f}[/bold green] "
            f"(or {results.effect_per_million * 100:.2f} percentage points)."
        )
        console.print(interpretation)
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

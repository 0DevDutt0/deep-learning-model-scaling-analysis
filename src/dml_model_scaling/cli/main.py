"""Main CLI application entry point."""

import typer
from rich.console import Console

from dml_model_scaling import __version__

app = typer.Typer(
    name="dml-scale",
    help="DML Model Scaling: Causal inference analysis of neural network scaling laws",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"deep-learning-model-scaling-analysis version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """DML Model Scaling CLI."""
    pass


from dml_model_scaling.cli.analyze import app as analyze_app
from dml_model_scaling.cli.train import app as train_app

app.add_typer(train_app, name="train", help="Run training experiments")
app.add_typer(analyze_app, name="analyze", help="Run causal analysis")


if __name__ == "__main__":
    app()

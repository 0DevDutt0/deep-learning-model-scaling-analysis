"""Orchestrate multiple experiments across configurations."""

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from torch.utils.data import DataLoader

from dml_model_scaling.config.experiment_config import ExperimentConfig
from dml_model_scaling.experiments.data import create_data_loader, load_mnist_dataset
from dml_model_scaling.experiments.results import ExperimentResult, ResultsManager
from dml_model_scaling.experiments.trainer import train_and_evaluate
from dml_model_scaling.models.utils import count_parameters, get_model_by_name
from dml_model_scaling.utils.logging import get_logger
from dml_model_scaling.utils.random import set_random_seed

logger = get_logger(__name__)
console = Console()


class ExperimentRunner:
    """Orchestrates running multiple experiments with different configurations.

    Attributes:
        config: Experiment configuration.
        results_manager: Manager for storing results.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the experiment runner.

        Args:
            config: Configuration for experiments.
        """
        self.config = config
        self.results_manager = ResultsManager(config.output_path)
        set_random_seed(config.random_seed)

    def _load_datasets(self) -> tuple[DataLoader, DataLoader]:
        """Load MNIST train and test datasets.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        train_dataset = load_mnist_dataset(
            root=self.config.data_dir,
            train=True,
            download=True,
        )

        test_dataset = load_mnist_dataset(
            root=self.config.data_dir,
            train=False,
            download=True,
        )

        test_loader = create_data_loader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_dataset, test_loader

    def run(self) -> Path:
        """Run all experiments defined in the configuration.

        Returns:
            Path to the results CSV file.
        """
        console.print(f"\n[bold blue]Starting experiment suite[/bold blue]")
        console.print(f"Total experiments: {self.config.total_experiments()}")
        console.print(f"Output: {self.config.output_path}\n")

        self.results_manager.initialize_csv()
        train_dataset, test_loader = self._load_datasets()

        exp_id = 1

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Running experiments...",
                total=self.config.total_experiments(),
            )

            for model_name in self.config.model_names:
                for dataset_size in self.config.dataset_sizes:
                    train_loader = create_data_loader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        num_workers=0,
                        subset_size=dataset_size,
                    )

                    for epochs in self.config.epochs_list:
                        for lr in self.config.learning_rates:
                            progress.update(
                                task,
                                description=f"[cyan]Exp {exp_id}: {model_name} "
                                f"| data={dataset_size} | epochs={epochs} | lr={lr}",
                            )

                            model = get_model_by_name(model_name)
                            model_size = count_parameters(model)

                            accuracy, training_time = train_and_evaluate(
                                model=model,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                epochs=epochs,
                                learning_rate=lr,
                                device=self.config.device,
                            )

                            result = ExperimentResult(
                                experiment_id=exp_id,
                                model_name=model_name,
                                model_size=model_size,
                                dataset_size=dataset_size,
                                epochs=epochs,
                                learning_rate=lr,
                                batch_size=self.config.batch_size,
                                training_time=training_time,
                                accuracy=accuracy,
                            )

                            self.results_manager.save_result(result)

                            logger.info(
                                f"Completed experiment {exp_id}: "
                                f"accuracy={accuracy:.4f}, time={training_time:.2f}s"
                            )

                            exp_id += 1
                            progress.advance(task)

        console.print(f"\n[bold green]✓ All experiments completed![/bold green]")
        console.print(f"Results saved to: {self.config.output_path}\n")

        return self.config.output_path

"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from dml_model_scaling.exceptions import DataError
from dml_model_scaling.utils.logging import get_logger

logger = get_logger(__name__)


def get_mnist_transforms(train: bool = True) -> transforms.Compose:
    """Get standard MNIST transforms.

    Args:
        train: Whether to include training augmentations.

    Returns:
        Composed transforms for MNIST data.
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    return transforms.Compose(transform_list)


def load_mnist_dataset(
    root: Path,
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """Load MNIST dataset.

    Args:
        root: Root directory to store/load data.
        train: Whether to load training or test set.
        download: Whether to download if not present.

    Returns:
        MNIST dataset.

    Raises:
        DataError: If data loading fails.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        transform = get_mnist_transforms(train=train)

        dataset = datasets.MNIST(
            root=str(root),
            train=train,
            download=download,
            transform=transform,
        )

        logger.info(f"Loaded MNIST {'train' if train else 'test'} set: {len(dataset)} samples")
        return dataset

    except Exception as e:
        raise DataError(f"Failed to load MNIST dataset: {e}") from e


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    subset_size: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader from a dataset.

    Args:
        dataset: PyTorch dataset.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        subset_size: If provided, only use the first N samples.

    Returns:
        Configured DataLoader.

    Raises:
        DataError: If DataLoader creation fails.
    """
    try:
        if subset_size is not None:
            if subset_size > len(dataset):
                logger.warning(
                    f"Requested subset size {subset_size} exceeds dataset size "
                    f"{len(dataset)}, using full dataset"
                )
                subset_size = len(dataset)
            dataset = Subset(dataset, range(subset_size))
            logger.info(f"Using subset of {subset_size} samples")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    except Exception as e:
        raise DataError(f"Failed to create DataLoader: {e}") from e

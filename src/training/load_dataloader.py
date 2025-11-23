from torch.utils.data import DataLoader

from ..ingest.dataloader import get_cifar10_dataloaders


def load_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load the dataloaders of the CIFAR-10 dataset

    Args:
        config (dict): Dictionary containing data configuration parameters.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for train, validation and test set.
    """
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(config["data"])
    return train_loader, val_loader, test_loader

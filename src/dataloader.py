from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from loguru import logger


def get_flower_dataloaders(
    data_dir: str = "../data/raw",
    batch_size: int = 32,
    val_split: float = 0.2,
    img_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders for flower classification dataset.

    Args:
        data_dir (str, optional): _description_. Path where the data is stored. Defaults to "../data/raw".
        batch_size (int, optional): _description_. Batch size for the dataloaders. Defaults to 32.
        val_split (float, optional): _description_. Fraction of data to use for validation. Defaults to 0.2.
        img_size (int, optional): _description_. Size to which images are resized (length and width). Defaults to 128.
        num_workers (int, optional): _description_. Number of workers for data loading. Defaults to 2.

    Returns:
        tuple[DataLoader, DataLoader]: _description_. Train and validation dataloaders.
    """

    # Path to data directory
    data_dir = Path(data_dir)

    # Resize, convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225],  # ImageNet stds
            ),
        ]
    )

    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split dataset into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Create dataloaders for train and validation sets
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader

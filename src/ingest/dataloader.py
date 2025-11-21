from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset


def get_flower_dataloaders(config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for the Flower102 classification dataset.

    Args:
        data_dir (str, optional): _description_. Path where the data is stored. Defaults to "../data/raw".
        batch_size (int, optional): _description_. Batch size for the dataloaders. Defaults to 32.
        val_split (float, optional): _description_. Fraction of data to use for validation. Defaults to 0.2.
        img_size (int, optional): _description_. Size to which images are resized (length and width). Defaults to 128.
        num_workers (int, optional): _description_. Number of workers for data loading. Defaults to 2.

    Returns:
        tuple[DataLoader, DataLoader]: _description_. Train and validation dataloaders.
    """

    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    val_split = config["val_split"]
    img_size = config["img_size"]
    num_workers = config["num_workers"]

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Load all splits
    ds_train = datasets.Flowers102(root=data_dir, split="train", transform=transform)
    ds_val = datasets.Flowers102(root=data_dir, split="val", transform=transform)
    ds_test = datasets.Flowers102(root=data_dir, split="test", transform=transform)

    # Combine train + val for our own train/val split
    full_train_set = ConcatDataset([ds_train, ds_val])
    train_len = int(len(full_train_set) * val_split)
    real_train_len = len(full_train_set) - train_len

    train_ds, val_ds = random_split(full_train_set, [real_train_len, train_len])

    # Test set stays separate
    test_ds = ds_test

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

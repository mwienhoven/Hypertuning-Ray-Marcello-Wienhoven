from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_cifar10_dataloaders(config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for the CIFAR-10 dataset.
    """

    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    val_split = config.get("val_split", 0.2)
    test_split = config.get("test_split", 0.2)
    img_size = config["img_size"]
    num_workers = config["num_workers"]

    # CIFAR-10 specific normalization values
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ]
    )

    # CIFAR-10 has only train and test sets
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Determine sizes
    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # Train / validation split
    train_ds, val_ds = random_split(full_train, [n_train, n_val])

    # Optionally reduce test set using config test_split
    if test_split < 1.0:
        new_test_size = int(len(test_ds) * test_split)
        test_ds, _ = random_split(
            test_ds, [new_test_size, len(test_ds) - new_test_size]
        )

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

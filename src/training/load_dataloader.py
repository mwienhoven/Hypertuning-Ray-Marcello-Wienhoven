from ingest.dataloader import get_flower_dataloaders


def load_dataloaders(
    data_dir: str, batch_size: int, val_split: float, img_size: int, num_workers: int
):
    train_loader, val_loader, test_loader = get_flower_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        img_size=img_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader

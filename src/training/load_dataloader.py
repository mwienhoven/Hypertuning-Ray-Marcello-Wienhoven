from ..ingest.dataloader import get_flower_dataloaders


def load_dataloaders(config):
    train_loader, val_loader, test_loader = get_flower_dataloaders(config["data"])
    return train_loader, val_loader, test_loader

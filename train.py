import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from loguru import logger
from src.config import load_config
from src.models.cnn import CNN


def main():
    # --- Config ---
    config = load_config("config.toml")

    data_dir = Path(config["data"]["data_dir"])
    batch_size = config["data"]["batch_size"]
    val_split = config["data"]["val_split"]
    img_size = config["data"]["img_size"]
    num_workers = config["data"]["num_workers"]

    learning_rate = config["train"]["learning_rate"]
    epochs = config["train"]["epochs"]
    optimizer_name = config["train"]["optimizer"]

    filters = config["model"]["filters"]
    units1 = config["model"]["units1"]
    units2 = config["model"]["units2"]
    num_classes = config["model"]["num_classes"]

    # --- Dataset & Dataloader ---
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info(
        f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}"
    )

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (1, 3, img_size, img_size)
    model = CNN(
        filters=filters,
        units1=units1,
        units2=units2,
        input_size=input_size,
        num_classes=num_classes,
    )
    model.to(device)
    logger.info(f"Model initialized with {filters=} {units1=} {units2=} {num_classes=}")

    # --- Optimizer & Loss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=learning_rate)
        if optimizer_name.lower() == "adam"
        else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    )

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # Save model
    torch.save(model.state_dict(), "cnn_flowers.pth")
    logger.info("✅ Model saved as cnn_flowers.pth")


if __name__ == "__main__":
    # Fix multiprocessing issues on Windows
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()

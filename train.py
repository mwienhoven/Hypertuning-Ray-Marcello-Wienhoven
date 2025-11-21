import torch
import torch.optim as optim
from pathlib import Path
from loguru import logger
from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes
from mltrainer.imagemodels import CNNConfig, CNNblocks
from config import load_config
from src.data import get_flower_dataloaders


def main() -> None:
    # --- Load config ---
    cfg = load_config("config.toml")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    # --- Dataloaders ---
    train_loader, val_loader = get_flower_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=data_cfg["batch_size"],
        val_split=data_cfg["val_split"],
        img_size=data_cfg["img_size"],
        num_workers=data_cfg["num_workers"],
    )
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CNN Config ---
    cnn_config = CNNConfig(
        matrixshape=(data_cfg["img_size"], data_cfg["img_size"]),
        batchsize=data_cfg["batch_size"],
        input_channels=3,  # Flowers102 is RGB
        hidden=model_cfg["filters"],
        kernel_size=model_cfg.get("kernel_size", 3),
        maxpool=model_cfg.get("maxpool", 2),
        num_layers=model_cfg.get("num_layers", 3),
        num_classes=model_cfg["num_classes"],
    )

    # --- Model ---
    model = CNNblocks(cnn_config)
    model.to(device)
    logger.info(f"Model initialized with config: {cnn_config}")

    # --- Loss, optimizer, metrics ---
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam
    accuracy = metrics.Accuracy()

    # --- Trainer settings ---
    settings = TrainerSettings(
        epochs=train_cfg["epochs"],
        metrics=[accuracy],
        logdir=Path(train_cfg.get("logdir", "logs")),
        train_steps=None,  # None = full epoch
        valid_steps=None,
        reporttypes=[ReportTypes.TOML],
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optimizer,
        traindataloader=train_loader,
        validdataloader=val_loader,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )

    # --- Training loop ---
    trainer.loop()

    # --- Save final model ---
    save_path = Path("models") / "cnn_flowers102.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()  # Windows-safe
    main()

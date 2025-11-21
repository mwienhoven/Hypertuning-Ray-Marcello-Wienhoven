import torch
import torch.optim as optim
from pathlib import Path
from loguru import logger
from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes
from mltrainer.imagemodels import CNNConfig, CNNblocks
from src.config import load_config
from src.dataloader import get_flower_dataloaders


def main() -> None:
    # --- Load config ---
    cfg = load_config("config.toml")
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    log_cfg = cfg.get("logging", {})

    # --- Dataloaders ---
    train_loader, val_loader = get_flower_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=data_cfg["batch_size"],
        val_split=data_cfg["val_split"],
        img_size=data_cfg["img_size"],
        num_workers=data_cfg["num_workers"],
    )
    logger.info(
        f"🔢 Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
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
        dropout=model_cfg.get("dropout", 0.0),
    )

    # --- Model ---
    model = CNNblocks(cnn_config)
    model.to(device)
    logger.info(f"✅ Model initialized with config: {cnn_config}")

    # --- Loss, optimizer, metrics ---
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam
    accuracy = metrics.Accuracy()

    # --- Trainer settings ---
    logdir = Path(log_cfg.get("logdir", "logs"))
    checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "models"))
    save_interval = log_cfg.get("save_interval", 5)
    report_types = [
        getattr(ReportTypes, rt) if isinstance(rt, str) else rt
        for rt in log_cfg.get("report_types", ["TOML"])
    ]

    settings = TrainerSettings(
        epochs=train_cfg["epochs"],
        metrics=[accuracy],
        logdir=logdir,
        train_steps=None,
        valid_steps=None,
        save_interval=save_interval,
        reporttypes=report_types,
        checkpoint_dir=checkpoint_dir,
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
    final_model_path = checkpoint_dir / "cnn_flowers102_final.pt"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✅ Final model saved to {final_model_path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()  # Windows-safe
    main()

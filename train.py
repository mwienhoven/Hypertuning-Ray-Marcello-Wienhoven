from pathlib import Path
import torch
from loguru import logger
from src.training.mlflow_utils import setup_mlflow, log_model
from src.training.train_utils import create_trainer
from src.config import load_config
from src.training.load_dataloader import load_dataloaders
from mltrainer.imagemodels import CNNConfig, CNNblocks


def main() -> None:
    cfg = load_config("config.toml")
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    log_cfg = cfg.get("logging", {})

    setup_mlflow("flowers102_experiment")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # --- Dataloaders ---
    train_loader, val_loader, test_loader = load_dataloaders(cfg)
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}"
    )

    # --- Model ---
    cnn_config = CNNConfig(
        matrixshape=(data_cfg["img_size"], data_cfg["img_size"]),
        batchsize=data_cfg["batch_size"],
        input_channels=3,
        hidden=model_cfg["filters"],
        kernel_size=model_cfg.get("kernel_size", 3),
        maxpool=model_cfg.get("maxpool", 2),
        num_layers=model_cfg.get("num_layers", 3),
        num_classes=model_cfg["num_classes"],
    )
    model = CNNblocks(cnn_config)
    model.to(device)
    logger.info(f"Model initialized with config: {cnn_config}")

    # --- Trainer ---
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        log_cfg=log_cfg,
        device=device,
    )

    # --- Start training ---
    trainer.loop()

    # --- Save final model ---
    save_path = Path(log_cfg.get("checkpoint_dir", "models")) / "cnn_flowers102.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Model saved to {save_path}")

    # --- Log model to MLflow ---
    log_model(save_path)


if __name__ == "__main__":
    main()

from datetime import datetime
from pathlib import Path
import torch
from loguru import logger
import mlflow
import shutil
import toml

from src.training.mlflow_utils import setup_mlflow, log_model
from src.training.train_utils import create_trainer
from src.config import load_config
from src.training.load_dataloader import load_dataloaders
from mltrainer.imagemodels import CNNConfig, CNNblocks


def main() -> None:
    # ---- Load config ----
    cfg = load_config("config.toml")
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    log_cfg = cfg.get("logging", {})

    # ---- MLflow ----
    setup_mlflow(log_cfg.get("experiment_name", "CIFAR-10_experiment"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_logdir = Path(log_cfg.get("log_dir", "logs"))
    run_logdir = base_logdir / timestamp
    run_logdir.mkdir(parents=True, exist_ok=True)

    log_file = run_logdir / "run.log"
    logger.add(log_file, level="INFO", encoding="utf-8")
    logger.info(f"📄 Logging to {log_file}")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # ---- Load data ----
    train_loader, val_loader, test_loader = load_dataloaders(cfg)
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, "
        f"Val samples: {len(val_loader.dataset)}, "
        f"Test samples: {len(test_loader.dataset)}"
    )

    # ---- Check for active MLflow run ----
    if mlflow.active_run() is not None:
        logger.warning("⚠️ MLflow run already active. Closing previous run...")
        mlflow.end_run()

    # ---- Start MLflow run ----
    with mlflow.start_run():

        mlflow.set_tag("model", log_cfg.get("model", "unknown"))
        mlflow.set_tag("developer", log_cfg.get("developer", "unknown"))

        # Log hyperparameters
        mlflow.log_param("batch_size", data_cfg["batch_size"])
        for key, value in model_cfg.items():
            mlflow.log_param(key, value)

        # ---- Build model ----
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
        logger.info(f"Model initialized: {cnn_config}")

        # ---- Trainer ----
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            log_cfg=log_cfg,
            device=device,
            logdir=run_logdir,
        )

        # ---- Training loop ----
        trainer.loop()

        # ---- Save final model ----
        ckpt_dir = Path(log_cfg.get("checkpoint_dir", "models"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_path = ckpt_dir / f"{timestamp}_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"✅ Saved model at {model_path}")

        shutil.copy("config.toml", run_logdir / "settings.toml")

        # --- Sla model config op als model.toml ---
        model_toml_path = run_logdir / "model.toml"
        with open(model_toml_path, "w") as f:
            toml.dump(model_cfg, f)

        # Log as MLflow artifact
        mlflow.log_artifact(str(model_path), artifact_path="pytorch_models")

        # ---- Log final test loss ----
        mlflow.log_metric("test_loss", trainer.test_loss)

        # ---- Return-like behaviour of objective() ----
        result = {
            "loss": trainer.test_loss,
            "status": "ok",
        }
        logger.info(f"Objective result: {result}")

        # ---- Log model (optional MLflow) ----
        log_model(model_path)


if __name__ == "__main__":
    runs = 3  # Amount of times to run the training
    for i in range(1, runs + 1):
        logger.info(f"🚀 Starting run {i}/{runs}")
        main()

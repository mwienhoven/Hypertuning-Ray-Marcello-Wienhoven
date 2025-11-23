from datetime import datetime
from pathlib import Path
import torch
from loguru import logger
import mlflow
from train_utils import create_trainer
from src.ingest.dataloader import load_dataloaders
from src.training.model import build_model


def objective(
    params: dict, data_cfg: dict, model_cfg: dict, train_cfg: dict, log_cfg: dict
):
    """Train a model with given hyperparameters and log everything to MLflow."""

    modeldir = Path(log_cfg.get("checkpoint_dir", "models"))
    modeldir.mkdir(parents=True, exist_ok=True)

    batch_size = data_cfg["batch_size"]

    # --- Load data ---
    train_loader, val_loader = load_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=batch_size,
        val_split=data_cfg["val_split"],
        img_size=data_cfg["img_size"],
        num_workers=data_cfg["num_workers"],
    )

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run():
        mlflow.set_tag("model", log_cfg.get("experiment_name", "default_experiment"))
        mlflow.set_tag("dev", log_cfg.get("developer", "unknown"))
        mlflow.log_params(params)
        mlflow.log_param("batch_size", batch_size)

        # --- Build model ---
        model, cnn_config = build_model(
            img_size=data_cfg["img_size"],
            batch_size=batch_size,
            model_cfg={**model_cfg, **params},
            device=device,
        )

        # --- Trainer ---
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            log_cfg=log_cfg,
            device=device,
        )

        # --- Training ---
        trainer.loop()

        # --- Save final model ---
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        model_path = modeldir / f"{tag}_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model at {model_path}")

        mlflow.log_artifact(str(model_path), artifact_path="pytorch_models")

        return {"loss": trainer.test_loss, "status": "ok"}

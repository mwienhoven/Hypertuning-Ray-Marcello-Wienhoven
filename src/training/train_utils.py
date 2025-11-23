from pathlib import Path
import torch
from mltrainer import Trainer, TrainerSettings, metrics, ReportTypes


def create_trainer(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    train_cfg: dict,
    log_cfg: dict = None,
    device: str = "cpu",
    logdir: Path = None,
) -> Trainer:
    """Configure MLTrainer trainer with given model and loaders using config."""
    if log_cfg is None:
        log_cfg = {}

    if logdir is None:
        logdir = Path(log_cfg.get("logdir", "logs"))

    acc = metrics.Accuracy()

    # Haal steps uit config, zet 0 om naar None
    train_steps = train_cfg.get("train_steps", 0)
    valid_steps = train_cfg.get("valid_steps", 0)
    train_steps = None if train_steps == 0 else train_steps
    valid_steps = None if valid_steps == 0 else valid_steps

    settings = TrainerSettings(
        epochs=train_cfg["epochs"],
        metrics=[acc],
        logdir=logdir,
        train_steps=train_steps,
        valid_steps=valid_steps,
        reporttypes=[ReportTypes.TOML, ReportTypes.MLFLOW],
        checkpoint_dir=Path(log_cfg.get("checkpoint_dir", "models")),
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=train_loader,
        validdataloader=val_loader,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )
    return trainer

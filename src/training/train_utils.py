from pathlib import Path
import torch
from mltrainer import Trainer, TrainerSettings, metrics, ReportTypes
from torch.utils.data import DataLoader


def create_trainer(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: dict,
    log_cfg: dict = None,
    device: str = "cpu",
    logdir: Path = None,
) -> Trainer:
    """Create the trainer to train the model

    Args:
        model (torch.nn.Module): The model to be trained
        train_loader (DataLoader): Dataloader for training data
        val_loader (DataLoader): Dataloader for validation data
        train_cfg (dict): Training configuration parameters
        log_cfg (dict): Logging configuration parameters. Defaults to None.
        device (str): The type of device to use (CPU or GPU). Defaults to "cpu".
        logdir (Path): Path where the logs will be written. Defaults to None.

    Returns:
        Trainer: Trainer object for training the model
    """
    if log_cfg is None:
        log_cfg = {}

    if logdir is None:
        logdir = Path(log_cfg.get("logdir", "logs"))

    acc = metrics.Accuracy()

    train_steps = train_cfg.get("train_steps", 0)
    valid_steps = train_cfg.get("valid_steps", 0)
    train_steps = (
        None if (train_steps == 0 or train_steps is None) else int(train_steps)
    )
    valid_steps = (
        None if (valid_steps == 0 or valid_steps is None) else int(valid_steps)
    )

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

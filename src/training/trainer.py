import torch
import torch.optim as optim
from pathlib import Path
from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes


def create_trainer(
    model,
    train_loader,
    val_loader,
    train_cfg: dict,
    log_cfg: dict,
    device: torch.device,
) -> Trainer:
    accuracy = metrics.Accuracy()

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

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optim.Adam,
        traindataloader=train_loader,
        validdataloader=val_loader,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )

    return trainer

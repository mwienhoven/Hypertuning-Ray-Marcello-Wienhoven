from pathlib import Path
from typing import Dict

import torch
from loguru import logger
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

import os
import sys

from src.config import load_config
from src.training.load_dataloader import load_dataloaders

# using direct TrainerSettings/Trainer from mltrainer in this script
from mltrainer.imagemodels import CNNConfig, CNNblocks
from mltrainer import ReportTypes

NUM_SAMPLES = 20
MAX_EPOCHS = 10

# Zet Ray tijdelijke directory naar een kort pad
ray_temp_dir = Path("C:/ray_temp").resolve()
ray_temp_dir.mkdir(parents=True, exist_ok=True)
os.environ["RAY_TMPDIR"] = str(ray_temp_dir)


def train_ray(tune_config: Dict) -> None:
    # Load base config
    cfg = load_config("config.toml")
    data_cfg = cfg["data"].copy()
    model_cfg = cfg["model"].copy()
    train_cfg = cfg["train"].copy()
    log_cfg = cfg.get("logging", {})

    # Override model/data hyperparameters from Ray
    model_cfg.update({k: v for k, v in tune_config.items() if k in model_cfg})
    if "batch_size" in tune_config:
        data_cfg["batch_size"] = int(tune_config["batch_size"])

    # Load dataloaders
    train_loader, val_loader, test_loader = load_dataloaders({"data": data_cfg})

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # Build model using mltrainer imagemodels
    cnn_config = CNNConfig(
        matrixshape=(data_cfg["img_size"], data_cfg["img_size"]),
        batchsize=data_cfg["batch_size"],
        input_channels=3,
        hidden=model_cfg.get("filters", 64),
        kernel_size=model_cfg.get("kernel_size", 3),
        maxpool=model_cfg.get("maxpool", 2),
        num_layers=model_cfg.get("num_layers", 3),
        num_classes=model_cfg.get("num_classes", 10),
    )
    model = CNNblocks(cnn_config).to(device)

    # Prepare trainer using create_trainer but ensure Ray reporting
    # create_trainer wraps mltrainer.Trainer; we will build settings via create_trainer
    # but override report types by passing a modified log_cfg if necessary.
    # create_trainer currently uses TOML+MLFLOW by default, so construct TrainerSettings
    # directly to ensure ReportTypes.RAY is used.
    from mltrainer import TrainerSettings, Trainer, metrics

    acc = metrics.Accuracy()

    # Determine train/valid steps (only pass if positive integer)
    train_steps = train_cfg.get("train_steps", 0)
    valid_steps = train_cfg.get("valid_steps", 0)
    settings_kwargs = dict(
        epochs=train_cfg.get("epochs", MAX_EPOCHS),
        metrics=[acc],
        logdir=Path(log_cfg.get("logdir", "logs")),
        reporttypes=[ReportTypes.RAY],
        checkpoint_dir=Path(log_cfg.get("checkpoint_dir", "models")),
    )
    if isinstance(train_steps, int) and train_steps > 0:
        settings_kwargs["train_steps"] = int(train_steps)
    if isinstance(valid_steps, int) and valid_steps > 0:
        settings_kwargs["valid_steps"] = int(valid_steps)

    trainersettings = TrainerSettings(**settings_kwargs)

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=train_loader,
        validdataloader=val_loader,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=str(device),
    )

    # Run training loop (mltrainer will report to Ray if ReportTypes.RAY is set)
    trainer.loop()

    # Final report to Ray (in case mltrainer didn't send final metric)
    tune.report(test_loss=getattr(trainer, "test_loss", None))


if __name__ == "__main__":
    # Ensure Ray uses the exact python executable for worker processes.
    # This avoids URL-encoding / path-quoting issues on Windows paths containing '&' or other chars.
    os.environ.setdefault("RAY_PYTHON_EXECUTABLE", sys.executable)

    # Also set a short temp dir for Ray to avoid long paths with special characters
    # (we already set RAY_TMPDIR above, keep it consistent)
    os.environ.setdefault("RAY_TMPDIR", str(ray_temp_dir))

    ray.init()

    tune_dir = Path("logs/ray").resolve()
    search = HyperOptSearch(metric="test_loss", mode="min")
    scheduler = ASHAScheduler(
        metric="test_loss",
        mode="min",
        max_t=MAX_EPOCHS,
        grace_period=1,
        reduction_factor=2,
    )

    # Search space — keep names consistent with config.toml model keys
    config = {
        "filters": tune.choice([16, 32, 64, 128]),
        "units1": tune.randint(64, 256),
        "units2": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.5),
        "num_layers": tune.randint(2, 4),
        "kernel_size": tune.choice([3, 5]),
    }

    reporter = CLIReporter(
        metric_columns=["test_loss"]
    )  # mltrainer reports 'test_loss'

    analysis = tune.run(
        train_ray,
        config=config,
        # metric="test_loss",
        # mode="min",
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        num_samples=NUM_SAMPLES,
        search_alg=search,
        scheduler=scheduler,
        verbose=1,
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    ray.shutdown()

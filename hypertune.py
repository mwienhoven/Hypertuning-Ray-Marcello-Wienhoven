from pathlib import Path
from typing import Dict
import os
import sys

import torch
from loguru import logger
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from src.config import load_config
from src.training.load_dataloader import load_dataloaders
from mltrainer.imagemodels import CNNConfig, CNNblocks
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics

NUM_SAMPLES = 2
MAX_EPOCHS = 1

# Paths
BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.toml"
RAY_TMPDIR = Path("C:/ray_temp").resolve()
RAY_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ["RAY_TMPDIR"] = str(RAY_TMPDIR)
os.environ["RAY_PYTHON_EXECUTABLE"] = sys.executable


def train_ray(tune_config: Dict) -> None:
    # Load base config
    cfg = load_config(CONFIG_PATH)
    data_cfg = cfg["data"].copy()
    model_cfg = cfg["model"].copy()
    train_cfg = cfg["train"].copy()
    log_cfg = cfg.get("logging", {})

    # Override with hyperparameters from Ray
    for key, val in tune_config.items():
        if key in model_cfg:
            model_cfg[key] = val
        if key == "batch_size":
            data_cfg["batch_size"] = int(val)

    # Load dataloaders
    train_loader, val_loader, test_loader = load_dataloaders({"data": data_cfg})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # Build model with dropout
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

    # Trainer
    acc = metrics.Accuracy()
    settings_kwargs = dict(
        epochs=train_cfg.get("epochs", MAX_EPOCHS),
        metrics=[acc],
        logdir=Path(log_cfg.get("logdir", "logs")),
        reporttypes=[ReportTypes.RAY],  # send metrics to Ray
        checkpoint_dir=Path(log_cfg.get("checkpoint_dir", "models")),
        train_steps=train_cfg.get("train_steps", 0),
        valid_steps=train_cfg.get("valid_steps", 0),
    )
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

    # Run training
    trainer.loop()

    # Report final metric to Ray
    tune.report(test_loss=getattr(trainer, "test_loss", None))


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    search_space = {
        "filters": tune.choice([16, 32, 64, 128]),
        "units1": tune.randint(64, 256),
        "units2": tune.randint(16, 128),
        "num_layers": tune.randint(2, 4),
        "kernel_size": tune.choice([3, 5]),
    }

    search = HyperOptSearch(metric="test_loss", mode="min")
    scheduler = ASHAScheduler(
        metric="test_loss",
        mode="min",
        max_t=MAX_EPOCHS,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["test_loss"])

    tune_dir = BASE_DIR / "logs" / "ray"
    tune_dir.mkdir(parents=True, exist_ok=True)

    analysis = tune.run(
        train_ray,
        config=search_space,
        num_samples=NUM_SAMPLES,
        search_alg=search,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        verbose=1,
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
    )

    print("✅ Best hyperparameters found:", analysis.best_config)
    ray.shutdown()

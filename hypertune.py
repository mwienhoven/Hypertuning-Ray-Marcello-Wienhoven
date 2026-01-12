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
from src.models.cnn import CNN
from mltrainer import Trainer, TrainerSettings, metrics

# -------------------------------
# GLOBAL SETTINGS
# -------------------------------
NUM_SAMPLES = 200  # Number of Ray Tune trials
MAX_EPOCHS = 10  # Max epochs per trial
RAY_TMPDIR = Path("C:/ray_temp").resolve()

RAY_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ["RAY_TMPDIR"] = str(RAY_TMPDIR)
os.environ["RAY_PYTHON_EXECUTABLE"] = sys.executable

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.toml"


# -------------------------------
# TRAIN FUNCTION USED BY RAY TUNE
# -------------------------------
def train_ray(tune_config: Dict) -> None:
    # Load base configuration
    cfg = load_config(CONFIG_PATH)

    # Copy sections so Ray can override them
    data_cfg = cfg["data"].copy()
    model_cfg = cfg["model"].copy()
    train_cfg = cfg["train"].copy()
    log_cfg = cfg.get("logging", {})

    # Override model params with Ray Tune suggestions
    for key, val in tune_config.items():
        if key in model_cfg:
            model_cfg[key] = val
        if key == "batch_size":
            data_cfg["batch_size"] = int(val)

    # Load dataloaders
    train_loader, val_loader, test_loader = load_dataloaders({"data": data_cfg})
    trainsteps = len(train_loader)
    validsteps = len(val_loader)
    trainstreamer = iter(train_loader)
    validstreamer = iter(val_loader)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # -------------------------------
    # BUILD MODEL
    # -------------------------------
    model = CNN(
        filters=model_cfg.get("filters", 64),
        units1=model_cfg.get("units1", 128),
        units2=model_cfg.get("units2", 64),
        input_size=(
            data_cfg["batch_size"],
            3,
            data_cfg["img_size"],
            data_cfg["img_size"],
        ),
        num_classes=model_cfg.get("num_classes", 10),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3))
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = metrics.Accuracy()

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for _ in range(train_cfg.get("epochs", MAX_EPOCHS)):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # -------------------------------
        # VALIDATION LOOP
        # -------------------------------
        model.eval()
        valid_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                valid_loss += loss.item()
                all_preds.append(outputs)
                all_labels.append(y)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Accuracy computation (safe fallback)
        try:
            accuracy = acc_metric(all_preds, all_labels)
        except Exception:
            if all_preds.ndim == 2:
                preds = all_preds.argmax(dim=1)
            else:
                preds = all_preds.long()
            preds = preds.cpu()
            labels = all_labels.cpu()
            accuracy = (preds == labels).float().mean().item()

        # -------------------------------
        # REPORT TO RAY TUNE
        # -------------------------------
        tune.report(
            {
                "train_loss": train_loss / trainsteps,
                "valid_loss": valid_loss / validsteps,
                "accuracy": accuracy,
            }
        )


# -------------------------------
# MAIN HYPERPARAMETER TUNING
# -------------------------------
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # -------------------------------
    # SEARCH SPACE
    # -------------------------------
    search_space = {
        "filters": tune.choice([16, 32, 64, 128]),
        "units1": tune.choice([64, 128, 256]),
        "units2": tune.choice([32, 64, 128]),
        "num_layers": tune.randint(2, 4),
        "kernel_size": tune.choice([3, 5]),
        "dropout": tune.uniform(0.0, 0.5),
    }

    search = HyperOptSearch(metric="valid_loss", mode="min")

    scheduler = ASHAScheduler(
        metric="valid_loss",
        mode="min",
        max_t=MAX_EPOCHS,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["valid_loss", "accuracy"])

    tune_dir = BASE_DIR / "logs" / "ray"
    tune_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # RUN RAY TUNE
    # -------------------------------
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

    best_config = analysis.get_best_config(metric="valid_loss", mode="min")
    print("Best hyperparameters found:", best_config)

    ray.shutdown()

from pathlib import Path
import mlflow
from loguru import logger


def setup_mlflow(experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
    """Set up MLflow experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment '{experiment_name}' set with URI '{tracking_uri}'")


def log_model(model_path: Path, artifact_path: str = "pytorch_models"):
    """Log a model artifact to MLflow."""
    mlflow.log_artifact(str(model_path), artifact_path)
    logger.info(f"Logged model {model_path} to MLflow under {artifact_path}")

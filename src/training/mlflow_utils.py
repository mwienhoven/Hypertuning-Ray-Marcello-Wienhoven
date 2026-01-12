from pathlib import Path
import mlflow
from loguru import logger


def setup_mlflow(experiment_name: str, tracking_uri: str | None = None) -> None:
    """Setup the mlflow environment

    Args:
        experiment_name (str): Name of the MLflow experiment.
        tracking_uri (str): Tracking URI for MLflow. Defaults to "sqlite:///mlflow.db".
    """

    if tracking_uri is None:
        tracking_path = Path("mlflow.db").resolve()
        tracking_uri = f"sqlite:///{tracking_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment '{experiment_name}' set with URI '{tracking_uri}'")


def log_model(model_path: Path, artifact_path: str = "pytorch_models") -> None:
    """Logging of the model in MLflow

    Args:
        model_path (Path): Path to the model file to be logged.
        artifact_path (str): Path in MLflow where the model will be stored. Defaults to "pytorch_models".
    """
    mlflow.log_artifact(str(model_path), artifact_path)
    logger.info(f"Logged model {model_path} to MLflow under {artifact_path}")

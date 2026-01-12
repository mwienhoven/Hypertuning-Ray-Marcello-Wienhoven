import torch
from mltrainer.imagemodels import CNNConfig, CNNblocks


def build_model(
    img_size: int, batch_size: int, model_cfg: dict, device: torch.device
) -> tuple[CNNblocks, CNNConfig]:
    """Building the model with the configurations

    Args:
        img_size (int): Pixel size of the image
        batch_size (int): Batch size in which the samples are loaded
        model_cfg (dict): Configuration dictionary for the model
        device (torch.device): Type of device to use (CPU or GPU)

    Returns:
        tuple[CNNblocks, CNNConfig]: The built model and its configuration
    """
    cnn_config = CNNConfig(
        matrixshape=(img_size, img_size),
        batchsize=batch_size,
        input_channels=3,  # RGB
        filters=model_cfg["filters"],
        kernel_size=model_cfg.get("kernel_size", 3),
        maxpool=model_cfg.get("maxpool", 2),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.3),
        num_classes=model_cfg["num_classes"],
    )
    model = CNNblocks(cnn_config)
    model.to(device)
    return model, cnn_config

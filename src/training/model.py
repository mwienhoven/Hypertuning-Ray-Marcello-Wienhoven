import torch
from mltrainer.imagemodels import CNNConfig, CNNblocks


def build_model(
    img_size: int, batch_size: int, model_cfg: dict, device: torch.device
) -> tuple[CNNblocks, CNNConfig]:
    cnn_config = CNNConfig(
        matrixshape=(img_size, img_size),
        batchsize=batch_size,
        input_channels=3,  # RGB
        hidden=model_cfg["filters"],
        kernel_size=model_cfg.get("kernel_size", 3),
        maxpool=model_cfg.get("maxpool", 2),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.0),
        num_classes=model_cfg["num_classes"],
    )
    model = CNNblocks(cnn_config)
    model.to(device)
    return model, cnn_config

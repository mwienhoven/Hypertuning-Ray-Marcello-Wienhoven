import torch
from torch import nn
from loguru import logger


class CNN(nn.Module):
    def __init__(
        self,
        filters: int,
        units1: int,
        units2: int,
        input_size: tuple,
        num_classes: int = 102,
    ) -> None:
        super().__init__()
        self.in_channels = input_size[1]  # input_size: (batch, channels, height, width)
        self.filters = filters
        self.units1 = units1
        self.units2 = units2
        self.input_size = input_size

        # Input vector size is (batch_size, in_channels, height, width)

        # Convolutional block
        self.convs = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dynamically calculate the activation map size after convs
        activation_map_size = self._get_activation_map_size(input_size)
        logger.info(f"Activation map size after convs: {activation_map_size}")

        # Global average pooling to reduce to (batch, filters, 1, 1)
        self.global_pool = nn.AvgPool2d(activation_map_size)

        # Fully connected block
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, num_classes),
        )

    def _get_activation_map_size(self, input_size):
        # Dummy forward pass to calculate the size of conv output
        with torch.no_grad():
            x = torch.ones(input_size)
            x = self.convs(x)
        return x.shape[-2:]  # (H, W)

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x

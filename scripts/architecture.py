import torch


class ResNet(torch.nn.Module):
    def __init__(self, n_in_channels: int = 2, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 3):
        super().__init__()

        self.initial_conv = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=n_kernels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.initial_relu = torch.nn.ReLU()

        residual_blocks = []
        for i in range(n_hidden_layers):
            residual_blocks.append(ResidualBlock(n_kernels, kernel_size))
        self.residual_blocks = torch.nn.Sequential(*residual_blocks)

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_kernels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        x = x.float()
        x = self.initial_conv(x)
        x = self.initial_relu(x)

        x = self.residual_blocks(x)

        predictions = self.output_layer(x)
        return predictions


class ResidualBlock(torch.nn.Module):
    def __init__(self, n_channels: int, kernel_size: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x
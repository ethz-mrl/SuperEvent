import torch
from torch import nn

class VggBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activate=True):
        super().__init__()
        self.activate = activate
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(output_channels)
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layers(x)
        if self.activate:
            x = self.activation(x)
        return x

class VggTransposeBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activate=True):
        super().__init__()
        self.activate = activate
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=1),  # padding of 1 preserves size
            nn.BatchNorm2d(output_channels)
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layers(x)
        if self.activate:
            x = self.activation(x)
        return x

class VggBackbone(nn.Module):
    def __init__(self, input_channels, output_channels, return_maxpool_indeces=False):
        super().__init__()
        self.return_maxpool_indeces = return_maxpool_indeces
        self.layers = nn.ModuleList([
            VggBlock(input_channels, 64, 3),
            VggBlock(64, 64, 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=return_maxpool_indeces),

            VggBlock(64, 64, 3),
            VggBlock(64, 64, 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=return_maxpool_indeces),

            VggBlock(64, 128, 3),
            VggBlock(128, 128, 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=return_maxpool_indeces),

            VggBlock(128, 128, 3),
            VggBlock(128, output_channels, 3)
        ])

    def forward(self, x):
        outputs = [x]
        maxpool_indeces = []
        for layer in self.layers:
            if self.return_maxpool_indeces and type(layer).__name__ == "MaxPool2d":
                out, ind = layer(outputs[-1])
                outputs.append(out)
                maxpool_indeces.append(ind)
            else:
                outputs.append(layer(outputs[-1]))
        return outputs, maxpool_indeces
    
class VggBackbone_Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MaxUnpool2d(kernel_size = 2, stride = 2),
            VggTransposeBlock(input_channels + 128, 128, 3),
            VggTransposeBlock(128, 64, 3),

            nn.MaxUnpool2d(kernel_size = 2, stride = 2),
            VggTransposeBlock(128, 64, 3),
            VggTransposeBlock(64, 64, 3),

            nn.MaxUnpool2d(kernel_size = 2, stride = 2),
            VggTransposeBlock(128, 128, 3),
            VggTransposeBlock(128, output_channels, 3)
        ])

    def forward(self, outputs, maxpool_indeces):
        maxpool_indeces_reversed_it = reversed(maxpool_indeces)
        down_outputs_reversed_it = reversed(outputs)
        skip_connection_in_next_layer = False

        # Skip results of last two layers
        next(down_outputs_reversed_it)
        next(down_outputs_reversed_it)

        x = outputs[-1]
        for layer in self.layers:
            output_down = next(down_outputs_reversed_it)
            if type(layer).__name__ == "MaxUnpool2d":
                x = layer(x, next(maxpool_indeces_reversed_it))
                skip_connection_in_next_layer = True
            else:
                if skip_connection_in_next_layer:
                    x = torch.concat((x, output_down), dim=1)  # skip connection
                    skip_connection_in_next_layer = False
                x = layer(x)
        return x
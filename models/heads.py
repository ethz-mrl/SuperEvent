import torch
from torch import nn

from models.backbones.vgg import VggBlock
from models.backbones.maxvit import MaxVitAttentionPairCl
from models.backbones.maxvit_backbone.layers.maxvit.maxvit import nhwC_2_nChw, nChw_2_nhwC

# For TensorRT, since 'prob = prob[:, :-1, :, :]' does not work
class RemoveLastChannel(nn.Module):
    def __init__(self, num_channels: int = 65):
        super().__init__()
        # register constant buffer with selected indices [0, ..., 63]
        self.register_buffer("indices", torch.arange(num_channels - 1))

    def forward(self, x):
        # x: (N, C, H, W), indices: (64,)
        return torch.index_select(x, dim=1, index=self.indices)

class DetectorHead(nn.Module):
    def __init__(self, config, input_channels=128, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.remove_dustbin = RemoveLastChannel(65)
        self.layers = nn.Sequential(
            VggBlock(input_channels, 256, 3),
            VggBlock(256, 1+pow(self.grid_size, 2), 1, activate=False)
            )

    def forward(self, x):
        x = self.layers(x)

        # PyTorch default is channels first [B, C, H, W]
        prob = torch.nn.functional.softmax(x, dim=1)

        # Remove dustbin for "no interest point"
        # prob = prob[:, :-1, :, :]
        # TensorRT fails on the line above
        prob = self.remove_dustbin(prob)  # (N, 64, H, W)

        prob = torch.nn.functional.pixel_shuffle(prob, self.grid_size)
        prob = torch.squeeze(prob, dim=1)

        return x, prob

class DetectorHeadFullRes(nn.Module):
    def __init__(self, config, input_channels=128):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            VggBlock(input_channels, 256, 3),
            VggBlock(256, 1, 1, activate=False)
            )

    def forward(self, x):
        x = self.layers(x)

        x = torch.squeeze(x, dim=1)
        prob = torch.nn.functional.sigmoid(x)

        return x, prob
    
class DescriptorHead(nn.Module):
    def __init__(self, config, input_channels=128, grid_size=8, descriptor_size=256, interpolate=True):
        super().__init__()
        self.grid_size = grid_size
        self.descriptor_size = descriptor_size
        self.interpolate = interpolate
        self.layers = nn.Sequential(
            VggBlock(input_channels, 256, 3),
            VggBlock(256, self.descriptor_size, 1, activate=False)
                )

    def forward(self, x):
        x = self.layers(x)

        if self.interpolate:
            # PyTorch default is channels first [B, C, H, W]
            desc_raw_shape = x.shape[2:]
            input_shape = [self.grid_size*desc_raw_shape[0], self.grid_size*desc_raw_shape[1]]
            desc = torch.nn.functional.interpolate(x, scale_factor=self.grid_size, mode="bilinear")
            # desc = torch.nn.functional.normalize(desc, dim=1)
            # safe, numerically stable version for TensorRT
            norm = (desc * desc).sum(dim=1, keepdim=True).sqrt()
            desc = desc / (norm + 1e-6)
        else:
            desc = torch.nn.functional.normalize(x, dim=1)

        return x, desc
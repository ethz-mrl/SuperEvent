"""
Original Yolox PAFPN code with slight modifications
"""
from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from models.backbones.maxvit_backbone.yolox_network_blocks import BaseConv, CSPLayer, DWConv
from models.backbones.maxvit import BackboneFeatures


class YOLOPAFPN(nn.Module):
    """
    Removed the direct dependency on the backbone.
    """

    def __init__(
            self,
            config,
            depth: float = 1.0,
            in_stages: Tuple[int, ...] = (2, 3, 4),
            in_channels: Tuple[int, ...] = (256, 512, 1024),
            depthwise: bool = False,
            act: str = "silu",
            compile_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.config = config
        assert len(in_stages) == len(in_channels)
        self.in_features = in_stages
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        ###### Compile if requested ######
        if compile_cfg is not None:
            compile_mdl = compile_cfg['enable']
            if compile_mdl and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg['args'])
            elif compile_mdl:
                print('Could not compile PAFPN because torch.compile is not available')

        ##################################

        # 'nearest-exact' not supported by ONNX, results are the same with 'nearest'
        self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        self.reduce_conv = nn.ModuleList()
        self.C3_p = nn.ModuleList()

        for i in range(len(in_channels) - 1):
            # For pixel-wise predictions, keep channels constant to satisfy required output channels when required
            if config["pixel_wise_predictions"] and in_channels[-i-1] < config["backbone_output_channels"] and i > 0:
                reduce_conv_channels_in = config["backbone_output_channels"]
                reduce_conv_channels_out = config["backbone_output_channels"] - in_channels[-i-2]
                C3_p_channels_in = config["backbone_output_channels"]
                C3_p_channels_out = config["backbone_output_channels"]
            elif config["pixel_wise_predictions"] and in_channels[-i-2] < config["backbone_output_channels"]:
                reduce_conv_channels_in = in_channels[-i-1]
                reduce_conv_channels_out = config["backbone_output_channels"] - in_channels[-i-2]
                C3_p_channels_in = config["backbone_output_channels"]
                C3_p_channels_out = config["backbone_output_channels"]
            else:
                reduce_conv_channels_in = in_channels[-i-1]
                reduce_conv_channels_out = in_channels[-i-2]
                C3_p_channels_in = 2 * in_channels[-i-2]
                C3_p_channels_out = in_channels[-i-2]
            self.reduce_conv.append(BaseConv(
                reduce_conv_channels_in, reduce_conv_channels_out, 1, 1, act=act
            ))
            self.C3_p.append(CSPLayer(
                C3_p_channels_in,
                C3_p_channels_out,
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            ))  # cat

        if config["pixel_wise_predictions"]:
            # Add one more layer for upsampling to full resolution
            self.reduce_conv.append(BaseConv(
                    config["backbone_output_channels"], config["backbone_output_channels"], 1, 1, act=act
                ))
            self.C3_p.append(CSPLayer(
                config["backbone_output_channels"],
                config["backbone_output_channels"],
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            ))  # cat

        else:
            # bottom-up conv
            self.bu_conv = nn.ModuleList()
            self.C3_n = nn.ModuleList()
            for i in range(len(in_channels) - 1):
                self.bu_conv.append(Conv(
                    in_channels[i], in_channels[i], 3, 2, act=act
                ))
                self.C3_n.append(CSPLayer(
                    2 * in_channels[i],
                    in_channels[i + 1],
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act,
                ))

                if in_channels[i + 1] == config["backbone_output_channels"]:
                    break

        ###### Compile if requested ######
        if compile_cfg is not None:
            compile_mdl = compile_cfg['enable']
            if compile_mdl and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg['args'])
            elif compile_mdl:
                print('Could not compile PAFPN because torch.compile is not available')
        ##################################

    def forward(self, input: BackboneFeatures):
        """
        Args:
            inputs: Feature maps from backbone

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        x = [input[f] for f in self.in_features]

        fpn_out = []
        f_out = x[-1]
        for i in range(len(self.C3_p)):
            if i < len(x) - 1:
                fpn_out.append(self.reduce_conv[i](f_out))
                f_out = self.upsample(fpn_out[i])
                f_out = th.cat([f_out, x[-i-2]], 1)
                f_out = self.C3_p[i](f_out)
            else:
                assert self.config["pixel_wise_predictions"]
                # Skip concat for final upsampling
                fpn_out.append(self.reduce_conv[i](f_out))
                f_out = self.upsample(fpn_out[i])
                f_out = self.C3_p[i](f_out)
                return f_out

        # Grid based prediction
        pan_out = f_out
        for i in range(len(self.C3_n)):
            p_out = self.bu_conv[i](pan_out) 
            p_out = th.cat([p_out, fpn_out[-i-1]], 1) 
            pan_out = self.C3_n[i](p_out)
        
        return pan_out
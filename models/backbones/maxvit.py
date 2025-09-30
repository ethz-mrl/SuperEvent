"""
The code in this file and in the maxvit_backbone folder is from the repo:
https://github.com/uzh-rpg/RVT which belongs to Mathias Gehrig and Davide Scaramuzza's
paper "Recurrent Vision Transformers for Object Detection with Event Cameras" (CVPR2023).
"""

from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from models.backbones.maxvit_backbone.layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)

FeatureMap = th.Tensor
BackboneFeatures = Dict[int, th.Tensor]

class MaxViTBackbone(nn.Module):
    def __init__(self, mdl_config: dict):
        super().__init__()

        ###### Config ######
        in_channels = mdl_config["input_channels"]
        embed_dim = mdl_config["embed_dim"]
        dim_multiplier_per_stage = tuple(mdl_config["dim_multiplier"])
        num_blocks_per_stage = tuple(mdl_config["num_blocks"])
        T_max_chrono_init_per_stage = tuple(mdl_config["T_max_chrono_init"])
        enable_masking = mdl_config["enable_masking"]

        num_stages = len(num_blocks_per_stage)
        #assert num_stages == 4

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)

        input_dim = in_channels
        patch_size = mdl_config["stem"]["patch_size"]
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]

        self.stages = nn.ModuleList()
        self.strides = []
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            stage = MaxViTBackboneStage(dim_in=input_dim,
                                     stage_dim=stage_dim,
                                     spatial_downsample_factor=spatial_downsample_factor,
                                     num_blocks=num_blocks,
                                     enable_token_masking=enable_masking_in_stage,
                                     T_max_chrono_init=T_max_chrono_init_stage,
                                     stage_cfg=mdl_config["stage"])
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            input_dim = stage_dim
            self.stages.append(stage)

        self.num_stages = num_stages

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)

    def forward(self, x: th.Tensor,
            token_mask: Optional[th.Tensor] = None) \
            -> BackboneFeatures:
        output: Dict[int, FeatureMap] = {}
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, token_mask if stage_idx == 0 else None)
            stage_number = stage_idx + 1
            output[stage_number] = x
        return output


class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim: int,
                 skip_first_norm: bool,
                 attention_cfg: dict):
        super().__init__()

        self.att_window = PartitionAttentionCl(dim=dim,
                                               partition_type=PartitionType.WINDOW,
                                               attention_cfg=attention_cfg,
                                               skip_first_norm=skip_first_norm)
        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type=PartitionType.GRID,
                                             attention_cfg=attention_cfg,
                                             skip_first_norm=False)

    def forward(self, x):
        x = self.att_window(x)
        x = self.att_grid(x)
        return x


class MaxViTBackboneStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: dict):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg["downsample"]
        attention_cfg = stage_cfg["attention"]

        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)

        ###### Mask Token ################
        self.mask_token = nn.Parameter(th.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)
        ##################################

    def forward(self, x: th.Tensor,
                token_mask: Optional[th.Tensor] = None) \
            -> FeatureMap:
        x = self.downsample_cf2cl(x)  # N C H W -> N H W C
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token
        for blk in self.att_blocks:
            x = blk(x)
        x = nhwC_2_nChw(x)
        return x
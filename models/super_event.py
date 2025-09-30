import numpy as np
from torch import nn

from models.backbones.vgg import VggBackbone, VggBackbone_Upsample
from models.backbones.maxvit import MaxViTBackbone
from models.backbones.maxvit_backbone.yolo_pafpn import YOLOPAFPN
from models.heads import DetectorHead, DetectorHeadFullRes, DescriptorHead

class SuperEvent(nn.Module):
    def __init__(self, config, tracing=False):
        super().__init__()
        self.backbone_type = config["backbone"]
        self.tracing = tracing

        if config["backbone"] == "vgg":
            self.backbone = VggBackbone(input_channels=config["input_channels"], output_channels=config["feature_channels"])
        elif config["backbone"] == "maxvit":
            self.backbone = MaxViTBackbone(config["backbone_config"])
            self.fpn = YOLOPAFPN(config,
                                 depth=config["backbone_config"]["fpn"]["depth"], 
                                 in_stages=np.array(config["backbone_config"]["fpn"]["in_stages"])[-4:].tolist(),
                                 in_channels=(config["backbone_config"]["embed_dim"] * np.array(config["backbone_config"]["dim_multiplier"])[-4:]).tolist(),
                                 depthwise=config["backbone_config"]["fpn"]["depthwise"],
                                 act=config["backbone_config"]["fpn"]["act"])
        else:
            raise NotImplementedError("Backbone", config["backbone"], " is not supported.")

        self.detector = DetectorHead(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], config=config)
        self.descriptor = DescriptorHead(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], descriptor_size=config["descriptor_size"], config=config)

    def forward(self, x):
        if self.backbone_type == "vgg":
            features = self.backbone(x)[0][-1]
        elif self.backbone_type == "maxvit":
            backbone_features = self.backbone(x)
            features = self.fpn(backbone_features)

        logits, prob = self.detector(features)
        descriptors_raw, descriptors = self.descriptor(features)

        if self.tracing:
            return prob, descriptors
        else:
            return {"logits": logits,
                    "prob": prob, 
                    "descriptors_raw": descriptors_raw, 
                    "descriptors": descriptors}
    
class SuperEventFullRes(nn.Module):
    def __init__(self, config, tracing=False):
        super().__init__()
        self.config = config
        self.tracing = tracing

        if config["backbone"] == "vgg":
            self.backbone_down = VggBackbone(input_channels=config["input_channels"], output_channels=config["feature_channels"], return_maxpool_indeces=True)
            self.backbone_up = VggBackbone_Upsample(input_channels=config["feature_channels"], output_channels=config["backbone_output_channels"])
        elif config["backbone"] == "maxvit":
            self.backbone = MaxViTBackbone(config["backbone_config"])
            self.fpn = YOLOPAFPN(config,
                                 depth=config["backbone_config"]["fpn"]["depth"], 
                                 in_stages=config["backbone_config"]["fpn"]["in_stages"],
                                 in_channels=config["backbone_config"]["embed_dim"] * np.array(config["backbone_config"]["dim_multiplier"]),
                                 depthwise=config["backbone_config"]["fpn"]["depthwise"],
                                 act=config["backbone_config"]["fpn"]["act"])
        else:
            raise NotImplementedError("Backbone", config["backbone"], " is not supported.")

        self.detector = DetectorHeadFullRes(input_channels=config["backbone_output_channels"], config=config)
        self.descriptor = DescriptorHead(input_channels=config["backbone_output_channels"], grid_size=1, descriptor_size=config["descriptor_size"], interpolate=False, config=config)

    def forward(self, x):
        if self.backbone_type == "vgg":
            features_compressed, maxpool_indeces = self.backbone_down(x)
            features = self.backbone_up(features_compressed, maxpool_indeces)
        elif self.backbone_type == "maxvit":
            backbone_features = self.backbone(x)
            features = self.fpn(backbone_features)

        logits, prob = self.detector(features)
        descriptors_raw, descriptors = self.descriptor(features)

        if self.tracing:
            return prob, descriptors
        else:
            return {"logits": logits,
                    "prob": prob, 
                    "descriptors_raw": descriptors_raw, 
                    "descriptors": descriptors}
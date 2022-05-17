import torch
import torch.nn as nn

from .utils import create_pointnet_components, create_mlp_components
from ..basemodel import BaseModel


class PVCNN(BaseModel):
    blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))
    n_output_point = 1024
    n_output_inst = 1024

    def __init__(self, width_multiplier=1, voxel_resolution_multiplier=1, **basemodel_kwargs):
        super().__init__(**basemodel_kwargs)
        self.in_channels = 3

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)
        bottleneck, _ = create_mlp_components(concat_channels_point + channels_point,
                                              out_channels=[512, 0.2, 512, self.n_output_point],
                                              dim=2, classifier=True, width_multiplier=width_multiplier)
        self.bottleneck = nn.Sequential(*bottleneck)

    def forward_features(self, inputs):
        features = inputs
        num_points = features.size(-1)
        coords = inputs
        out_features_list = []
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        return self.bottleneck(torch.cat(out_features_list, dim=1))

    def forward_instance(self, x, pooling='max'):
        return x.max(dim=2)[0]


class PVCNNSegmentation(nn.Module):
    def __init__(self, backbone, n_parts, n_classes, head='mlp'):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv1d(self.backbone.n_output_point, n_parts, 1)

    def forward(self, x, labels):
        x = self.backbone.forward_features(x)
        return self.head(x)


"""
DGCNN for part segmentation
from https://github.com/AnTao97/dgcnn.pytorch/blob/a42d68ac04f49f754f4dcc74817ddc37f7f4ee48/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .basemodel import BaseModel


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k, use_bn=True, double_mlp=False):
        super().__init__()
        self.k = k
        norm = nn.BatchNorm2d(out_channels) if use_bn else nn.GroupNorm(min(out_channels // 8, 16), out_channels)

        if double_mlp:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=True),
            )

    def knn(self, x):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

    def get_graph_feature(self, x, idx=None, dim9=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if not dim9:
                idx = self.knn(x)  # (batch_size, num_points, k)
            else:
                idx = self.knn(x[:, 6:])
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).expand(-1, -1, self.k, -1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

    def forward(self, x):
        x = self.get_graph_feature(x)
        return self.conv(x).max(dim=-1)[0]


class TransformNet(nn.Module):
    def __init__(self, use_bn=True):
        super().__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.GroupNorm(16, 64)
        self.bn2 = nn.BatchNorm2d(128) if use_bn else nn.GroupNorm(32, 128)
        self.bn3 = nn.BatchNorm1d(1024) if use_bn else nn.GroupNorm(32, 1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512) if use_bn else nn.GroupNorm(32, 512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256) if use_bn else nn.GroupNorm(32, 256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2, inplace=True)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2, inplace=True)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNNSegBackbone(BaseModel):
    n_output_inst = 1024 + 64 * 3
    n_output_point = 1024 + 64 * 3

    def __init__(self, use_bn=True, k=40, **basemodel_kwargs):
        super().__init__(**basemodel_kwargs)
        # self.transform_net = TransformNet(use_bn)
        self.conv1 = EdgeConv(3, 64, k, double_mlp=True, use_bn=use_bn)
        self.conv2 = EdgeConv(64, 64, k, double_mlp=True, use_bn=use_bn)
        self.conv3 = EdgeConv(64, 64, k, use_bn=use_bn)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(64 * 3, 1024, 1, bias=False),
            nn.BatchNorm1d(1024) if use_bn else nn.GroupNorm(32, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward_features(self, x, return_intermediate=False, as_backbone=False):
        # x0 = self.conv1.get_graph_feature(x)
        # t = self.transform_net(x0)
        # x = torch.bmm(x.transpose(2, 1), t)
        # x = x.transpose(2, 1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x = self.mlp1(torch.cat((x1, x2, x3), dim=1)).max(dim=-1, keepdim=True)[0]
        x = torch.cat((x.expand(-1, -1, x1.size(2)), x1, x2, x3), dim=1)

        if as_backbone:
            return x

        if return_intermediate:
            return x, (x1, x2, x3)

        return x

    def forward_instance(self, features, pooling='mean'):
        if pooling == 'mean':
            x = features.mean(dim=2)
        elif pooling == 'max':
            x = features.max(dim=2)[0]
        else:
            x = torch.cat((features.max(dim=2)[0], features.mean(dim=2)), dim=1)

        return x


class DGCNNClfBackbone(BaseModel):
    n_output_inst = 2048
    n_output_point = 1024

    def __init__(self, use_bn=True, k=40, **basemodel_kwargs):
        super().__init__(**basemodel_kwargs)
        self.conv1 = EdgeConv(3, 64, k, use_bn=use_bn)
        self.conv2 = EdgeConv(64, 64, k, use_bn=use_bn)
        self.conv3 = EdgeConv(64, 128, k, use_bn=use_bn)
        self.conv4 = EdgeConv(128, 256, k, use_bn=use_bn)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward_features(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return self.mlp1(x)

    def forward_instance(self, features, pooling='max_mean'):
        if pooling == 'mean':
            x = features.mean(dim=2)
        elif pooling == 'max':
            x = features.max(dim=2)[0]
        else:
            x = torch.cat((features.max(dim=2)[0], features.mean(dim=2)), dim=1)
        return x


class DGCNNSegmentation(nn.Module):
    def __init__(self, backbone, n_parts, n_classes, head='mlp'):
        super(DGCNNSegmentation, self).__init__()
        self.backbone = backbone

        self.category_embedding = nn.Sequential(
            nn.Linear(n_classes, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Conv1d(1024 + 64 * 4, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(256, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(256, 128, 1, bias=False),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(128, n_parts, 1, bias=False)
            )
        else:
            self.head = nn.Conv1d(1024 + 64 * 4, n_parts, 1)

    def forward(self, x, labels):
        x = self.backbone.forward_features(x, as_backbone=True).transpose(2, 1).contiguous().view(-1, self.backbone.n_output_point)
        x = self.backbone.mlp(x, idx=4).view(labels.size(0), -1, x.size(1)).transpose(2, 1)
        labels = self.category_embedding(labels).unsqueeze(2).expand(-1, -1, x.size(2))
        return self.head(torch.cat((x, labels), dim=1))


class DGCNNClassification(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.n_output_inst, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_instance(x)
        return self.head(x)

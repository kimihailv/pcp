import torch
import torch.nn as nn
from .basemodel import BaseModel


class TransformNet(nn.Module):
    def __init__(self, in_channels, regularizer=False):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, in_channels**2)
        )

        self.in_channels = in_channels
        self.regularizer = regularizer

    def forward(self, x):
        t = self.mlp1(x).max(dim=2)[0]
        t = self.mlp2(t).view(-1, self.in_channels, self.in_channels)
        t += torch.eye(self.in_channels, device=x.device).unsqueeze(0)

        x = torch.bmm(x.transpose(2, 1), t).transpose(2, 1).contiguous()

        if self.regularizer:
            return x, self.compute_regularizer(t)

        return x

    def compute_regularizer(self, t, coef=1e-3):
        t = torch.bmm(t, t.transpose(2, 1))
        identity = torch.eye(self.in_channels, device=t.device).unsqueeze(0)
        return coef * (t - identity).pow(2).sum(dim=(1, 2)).div(2).mean()


class PointNet(BaseModel):
    n_output_point = 1024 + 64
    n_output_inst = 1024 + 64

    def __init__(self, **basemodel_kwargs):
        super().__init__(**basemodel_kwargs)
        self.tn3 = TransformNet(3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.tn64 = TransformNet(64, True)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        self.reg = 0

    def forward_features(self, x):
        x = self.tn3(x)
        x = self.mlp1(x)
        x1, self.reg = self.tn64(x)
        global_emb = self.mlp2(x1).max(dim=2, keepdim=True)[0]
        x1 = torch.cat((x1, global_emb.expand(-1, -1, x.size(2))), dim=1)

        return x1

    def forward_instance(self, features, pooling='max'):
        return features.max(dim=2)[0]


class PointNetClassification(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(self.backbone.n_output_inst, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        x = self.backbone.forward_instance(features)
        return self.head(x)


class PointNetSeg(nn.Module):
    def __init__(self, encoder, n_parts, n_classes, head='linear'):
        super().__init__()
        self.backbone = encoder
        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Conv1d(self.backbone.n_output_point, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Conv1d(128, n_parts, 1)
            )
        else:
            self.head = nn.Conv1d(self.backbone.n_output_point, n_parts, 1)

    def forward(self, x, labels):
        batch_size, _, n_points = x.shape
        x = self.backbone.forward_features(x)
        x = self.apply_projector(x, layer_idx=4)
        return self.head(x)

    def apply_projector(self, x, layer_idx):
        batch_size, dim, n_points = x.shape
        x = x.transpose(2, 1).contiguous().view(-1, dim)
        x = self.backbone.mlp(x, idx=layer_idx).contiguous().view(batch_size, n_points, -1).transpose(2, 1)
        return x

import torch.nn as nn
import torch
from .encoder import Encoder
from .modules import PointNetMSG, FeaturePropagation, Attention1d, AdaGN, DownsampleBlock


class NoisePredictor1(nn.Module):
    def __init__(self, latent_size, time_dim):
        super().__init__()
        self.encoder = Encoder(latent_size)
        ctx_channels = latent_size + time_dim
        self.an0 = AdaGN(3, 64, 8, ctx_channels)

        '''self.down1 = PointNetMSG(n_centroids=1024,
                                 max_n_points=[128, 256],
                                 radius=[0.1, 0.3],
                                 in_channels=64 + 3,
                                 hid_channels=[[64, 64, 64], [64, 64, 64]])  # b x 128 x 1024'''
        self.down1 = DownsampleBlock(64 + 3, [64, 128, 128], 1024)
        self.an1 = AdaGN(128, 128, 8, ctx_channels)

        '''self.down2 = PointNetMSG(n_centroids=256,
                                 max_n_points=[64, 128],
                                 radius=[0.3, 0.5],
                                 in_channels=128 + 3,
                                 hid_channels=[[128, 256, 128], [128, 256, 128]])  # b x 256 x 256'''

        self.down2 = DownsampleBlock(128 + 3, [128, 256, 256], 256)
        self.an2 = AdaGN(256, 256, 16, ctx_channels)

        '''self.down3 = PointNetMSG(n_centroids=32,
                                 max_n_points=[16, 32],
                                 radius=[0.5, 0.7],
                                 in_channels=256 + 3,
                                 hid_channels=[[256, 512, 256], [256, 512, 256]])  # b x 512 x 32'''

        self.down3 = DownsampleBlock(256 + 3, [256, 512, 512], 32)
        self.an3 = AdaGN(512, 512, 32, ctx_channels)
        self.attn1 = Attention1d(512, 512, 512)

        self.up1 = FeaturePropagation(256, 512, [512, 256, 256])  # b x 256 x 256
        self.an4 = AdaGN(256, 256, 16, ctx_channels)

        self.up2 = FeaturePropagation(128, 256, [256, 256, 128])  # b x 128 x 1024
        self.an5 = AdaGN(128, 128, 8, ctx_channels)

        self.up3 = FeaturePropagation(64, 128, [256, 256, 256])  # b x 256 x 2048
        self.an6 = AdaGN(256, 256, 16, ctx_channels)

        self.predictor = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )

    def forward(self, x, xt, time_emb, return_features, z=None, deterministic_latent=False):
        mean = log_var = None

        if z is None:
            z, mean, log_var = self.get_latent(time_emb.size(0), time_emb.device, x, deterministic_latent)

        ctx = torch.cat((z, time_emb), dim=1)
        feat0 = self.an0(xt, ctx)

        xyz1, feat1 = self.down1(xt, point_features=feat0)
        feat1 = self.an1(feat1, ctx)

        xyz2, feat2 = self.down2(xyz1, point_features=feat1)
        feat2 = self.an2(feat2, ctx)

        xyz3, feat3 = self.down3(xyz2, point_features=feat2)
        feat3 = self.an3(feat3, ctx)
        feat3 = self.attn1(feat3, feat3)  # 32

        feat2 = self.up1(xyz2, xyz3, feat2, feat3)  # 256
        feat2 = self.an4(feat2, ctx)

        feat1 = self.up2(xyz1, xyz2, feat1, feat2)  # 1024
        feat1 = self.an5(feat1, ctx)

        feat0 = self.up3(xt, xyz1, feat0, feat1)  # 2048
        feat0 = self.an6(feat0, ctx)
        et = self.predictor(feat0)

        if not return_features:
            return et, (z, mean, log_var)

        features = [feat0, feat1, feat2, feat3]

        return features, [xt, xyz1, xyz2, xyz3], (et, z, mean, log_var)

    def get_latent(self, batch_size, device, x=None, deterministic=False):
        mean = log_var = None
        if x is not None:
            z, mean, log_var = self.encoder(x)
            if deterministic:
                z = mean
        else:
            # z = self.encoder.flow.sample(batch_size)
            z = torch.randn(batch_size, 512, device=device)

        return z, mean, log_var


class NoisePredictor(nn.Module):
    def __init__(self, latent_size, time_dim, residual=True):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.act = nn.functional.leaky_relu
        self.residual = residual
        self.latent_size = latent_size

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.BatchNorm1d(time_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.an1 = AdaGN(3, 128, 8, latent_size + time_dim) # 32
        self.an2 = AdaGN(128, 256, 16, latent_size + time_dim) # 32
        self.an3 = AdaGN(256, 512, 32, latent_size + time_dim) # 32
        # self.an4 = AdaGN(512, 256, 16, latent_size + time_dim) # 8
        # self.an5 = AdaGN(256, 128, 8, latent_size + time_dim) # 4
        # self.an6 = AdaGN(128, 3, 1, latent_size + time_dim) # 1

        # self.attn1 = Attention1d(128, 128, 128, 3)
        self.linear = nn.Conv1d(3, 3, 1)
        self.linear.weight.data = torch.eye(3).unsqueeze(2)
        self.linear.bias.data.zero_()

        self.mlp = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 3, 1)
        )

    def forward(self, x, xt, time_emb, return_features, z=None, deterministic_latent=False):
        mean, log_var = None, None

        if z is None:
            z, mean, log_var = self.get_latent(time_emb.size(0), time_emb.device, x, deterministic_latent)

        ctx_emb = torch.cat([self.time_embed(time_emb), z], dim=1)

        x1 = self.act(self.an1(xt, ctx_emb))
        x2 = self.act(self.an2(x1, ctx_emb))
        x3 = self.act(self.an3(x2, ctx_emb))
        # x4 = self.act(self.an4(x3, ctx_emb))
        # x4 = self.attn1(x4, x4)
        # x5 = self.act(self.an5(x4, ctx_emb))
        features = [x1, x2, x3]

        x3 = self.mlp(x3)
        # x6 = self.an6(x5, ctx_emb)

        if self.residual:
            et = self.linear(xt) + x3
        else:
            et = x3

        if return_features:
            return (features, None), (et, z, mean, log_var)

        return et, (z, mean, log_var)

    def get_latent(self, batch_size, device, x=None, deterministic=False):
        mean = log_var = None
        if x is not None:
            z, mean, log_var = self.encoder(x)
            if deterministic:
                z = mean
        else:
            z = self.encoder.flow.sample(batch_size)
            # z = torch.randn(batch_size, self.latent_size, device=device)

        return z, mean, log_var

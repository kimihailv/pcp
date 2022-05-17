import torch
import torch.nn as nn
from .modules import TransformNet, EdgeConv, AdaGN, MLP, Attention1d, PointNetMSG, FeaturePropagation


class Generator1(nn.Module):
    def __init__(self, style_size, style_emb_size, time_emb_size, use_bn=False):
        super().__init__()
        self.style_size = style_size
        self.time_emb_size = time_emb_size
        self.mapping_net = nn.Sequential(
            nn.Linear(style_size, style_emb_size),
            nn.SiLU(inplace=True),
            nn.Linear(style_emb_size, style_emb_size),
            nn.SiLU(inplace=True),
            nn.Linear(style_emb_size, style_emb_size),
            nn.SiLU(inplace=True)
        )

        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(time_emb_size, time_emb_size)
        )

        self.conv1 = EdgeConv(3, 64, 20, double_mlp=True, use_bn=use_bn)
        self.adagn1 = AdaGN(64, 64, 8, style_emb_size, use_linear=False)

        self.conv2 = EdgeConv(64, 128, 30, use_bn=use_bn)
        self.adagn2 = AdaGN(128, 128, 16, style_emb_size, use_linear=False)

        self.conv3 = EdgeConv(128, 256, 40, use_bn=use_bn)
        self.adagn3 = AdaGN(256, 256, 32, style_emb_size, use_linear=False)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(64 + 128 + 256 + time_emb_size, 1024, 1, bias=False),
            nn.BatchNorm1d(1024) if use_bn else nn.GroupNorm(64, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512) if use_bn else nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 256, 1)
        )
        local_features_size = 64 + 128 + 256 + 256
        self.attn1 = Attention1d(local_features_size, local_features_size, 512, 1)

        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 256, 1)
        )

        self.adagn4 = AdaGN(256, 256, 32, style_emb_size, use_linear=False)

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 128, 1)
        )

        self.attn2 = Attention1d(128, 128, 128, 1)

        self.adagn5 = AdaGN(128, 128, 16, style_emb_size, use_linear=False)

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 64, 1)
        )
        self.adagn6 = AdaGN(64, 64, 8, style_emb_size, use_linear=False)

        self.final_mlp = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 3, 1)
        )

        self.attn3 = Attention1d(3, 3, 3, 3)

    def forward(self, xt, z, time_emb):
        z = z / (z.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-7)
        z = self.mapping_net(z)

        time_emb = self.time_embed(time_emb).unsqueeze(2).expand(-1, -1, xt.size(2))

        x1 = self.adagn1(self.conv1(xt), z)
        x2 = self.adagn2(self.conv2(x1), z)
        x3 = self.adagn3(self.conv3(x2), z)

        x = self.mlp1(torch.cat((x1, x2, x3, time_emb), dim=1)).max(dim=-1, keepdim=True)[0]
        x = torch.cat((x.expand(-1, -1, x1.size(2)), x1, x2, x3), dim=1)
        x = self.attn1(x, x)

        x = self.adagn4(self.conv4(x), z)
        x = self.conv5(x)
        x = self.attn2(x, x)
        x = self.adagn5(x, z)
        x = self.adagn6(self.conv6(x), z)

        x = self.final_mlp(x)

        return xt + self.attn3(x, x)


class Generator(nn.Module):
    def __init__(self, n_patches, style_size, style_emb_size, time_emb_size, use_bn=False, n_points=2048):
        super().__init__()
        self.style_size = style_size
        self.time_emb_size = time_emb_size

        u = (torch.arange(0, 32) / 32 - 0.5).repeat(64)
        v = (torch.arange(0, 64) / 64 - 0.5).expand(32, -1).t().reshape(-1)
        grid = torch.stack((u, v), 1).t().unsqueeze(0)

        self.register_buffer('grid', grid)

        self.mapping_net = nn.Sequential(
            nn.Linear(style_size, style_emb_size),
            nn.SiLU(inplace=True),
            nn.Linear(style_emb_size, style_emb_size),
            nn.SiLU(inplace=True),
            nn.Linear(style_emb_size, style_emb_size),
            nn.SiLU(inplace=True)
        )

        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(time_emb_size, time_emb_size)
        )

        self.transform_net = TransformNet(use_bn)
        self.conv1 = EdgeConv(3, 64, 20, double_mlp=True, use_bn=use_bn)
        self.adagn1 = AdaGN(64, 64, 16, style_emb_size, use_linear=False)

        self.conv2 = EdgeConv(64, 64, 30, double_mlp=True, use_bn=use_bn)
        self.adagn2 = AdaGN(64, 64, 16, style_emb_size, use_linear=False)

        self.conv3 = EdgeConv(64, 64, 40, use_bn=use_bn)
        self.adagn3 = AdaGN(64, 64, 16, style_emb_size, use_linear=False)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(64 * 3 + time_emb_size, 512, 1, bias=False),
            nn.BatchNorm1d(512) if use_bn else nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(512 + 64 * 3 + time_emb_size, 1024, 1, bias=False),
            nn.BatchNorm1d(1024) if use_bn else nn.GroupNorm(64, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(1024, 512, 1, bias=False),
            nn.BatchNorm1d(512) if use_bn else nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, 1)
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512)
        )

        '''self.n_patches = n_patches
        self.points_per_patch = n_points // n_patches
        self.deforms = nn.ModuleList([MLP(3 + 512,
                                          [512, 512, 256],
                                          use_bn=use_bn) for _ in range(n_patches)])

        self.attentions = nn.ModuleList([Attention1d(256, 512, 256, 2) for _ in range(n_patches)])
        self.adagns = nn.ModuleList([AdaGN(256, 256, 32, style_emb_size + time_emb_size,
                                           use_linear=False) for _ in range(n_patches)])

        self.global_attention = Attention1d(256, 512, 512, 3)
        self.final = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256) if use_bn else nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128) if use_bn else nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 3, 1)
        )'''

        self.folding1 = nn.Sequential(
            nn.Conv1d(2 + 512 + time_emb_size, 1024, 1),
            nn.BatchNorm1d(1024) if use_bn else nn.GroupNorm(64, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512) if use_bn else nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 256, 1)
        )
        self.attn1 = Attention1d(256, 512, 256, 3)
        self.adagn_fold1 = AdaGN(256, 256, 32, style_emb_size, use_linear=False)

        self.folding2 = nn.Sequential(
            nn.Conv1d(256 + 512 + time_emb_size, 1024, 1),
            nn.BatchNorm1d(1024) if use_bn else nn.GroupNorm(64, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512) if use_bn else nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256) if use_bn else nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 3, 1),
        )

        self.attn2 = Attention1d(3, 512, 3, 3)

    def forward_features(self, x, z, time_emb):
        time_emb = self.time_embed(time_emb).unsqueeze(2).expand(-1, -1, x.size(2))
        x0 = self.conv1.get_graph_feature(x)
        t = self.transform_net(x0)
        x = torch.bmm(x.transpose(2, 1), t)
        x = x.transpose(2, 1)

        x1 = self.adagn1(self.conv1(x), z)
        x2 = self.adagn2(self.conv2(x1), z)
        x3 = self.adagn3(self.conv3(x2), z)

        x = self.mlp1(torch.cat((x1, x2, x3, time_emb), dim=1)).max(dim=-1, keepdim=True)[0]
        x = torch.cat((x.expand(-1, -1, x1.size(2)), x1, x2, x3, time_emb), dim=1)
        x = self.mlp2(x)

        return x

    def forward(self, x, z, time_emb):
        z = self.mapping_net(z)
        features = self.forward_features(x, z, time_emb)
        latent = torch.cat((features.max(dim=2)[0], features.mean(dim=2)), dim=1)
        latent = self.bottleneck(latent).unsqueeze(2).expand(-1, -1, x.size(2))
        # patches = []
        time_emb = self.time_embed(time_emb).unsqueeze(2).expand(-1, -1, x.size(2))
        # ctx = torch.cat((z, time_emb), dim=1)

        '''for i, (deform, attn, adagn) in enumerate(zip(self.deforms, self.attentions, self.adagns)):
            patch = torch.rand(x.size(0), 3, self.points_per_patch, device=latent.device)
            patch[:, 2:, :] = 0
            patch = torch.cat((patch, latent), dim=1)
            patch = deform(patch)
            patch = patch + attn(patch, features)
            patches.append(adagn(patch, ctx))

        patches = torch.cat(patches, dim=2)
        patches = self.global_attention(patches, features)
        return self.final(patches)'''

        x = torch.cat((self.grid.expand(x.size(0), -1, -1), latent, time_emb), dim=1)
        x = self.folding1(x)
        x = self.attn1(x, features)
        x = self.adagn_fold1(x, z)
        x = torch.cat((x, latent, time_emb), dim=1)
        x = self.folding2(x)

        return self.attn2(x, features)


class EncoderDecoder(nn.Module):
    def __init__(self, style_emb_size, time_emb_size, use_bn=False):
        super().__init__()
        ctx_channels = style_emb_size + time_emb_size
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(time_emb_size, time_emb_size)
        )
        self.an0 = AdaGN(3, 64, 8, ctx_channels)

        self.down1 = PointNetMSG(n_centroids=1024,
                                 max_n_points=[128, 256],
                                 radius=[0.1, 0.3],
                                 in_channels=64 + 3,
                                 hid_channels=[[64, 64, 64], [64, 64, 64]],
                                 use_bn=use_bn)  # b x 128 x 1024
        self.an1 = AdaGN(128, 128, 8, ctx_channels)

        self.down2 = PointNetMSG(n_centroids=256,
                                 max_n_points=[64, 128],
                                 radius=[0.3, 0.5],
                                 in_channels=128 + 3,
                                 hid_channels=[[128, 256, 128], [128, 256, 128]],
                                 use_bn=use_bn)  # b x 256 x 256

        self.an2 = AdaGN(256, 256, 16, ctx_channels)

        self.down3 = PointNetMSG(n_centroids=32,
                                 max_n_points=[16, 32],
                                 radius=[0.5, 0.7],
                                 in_channels=256 + 3,
                                 hid_channels=[[256, 512, 256], [256, 512, 256]],
                                 use_bn=use_bn)  # b x 512 x 32

        # self.attn1 = Attention1d(512, 512, 512)

        self.an3 = AdaGN(512, 512, 64, ctx_channels)

        self.up1 = FeaturePropagation(256, 512, [512, 256, 256], use_bn=use_bn)  # b x 256 x 256
        self.an4 = AdaGN(256, 256, 32, ctx_channels)

        self.up2 = FeaturePropagation(128, 256, [256, 256, 128], use_bn=use_bn)  # b x 128 x 1024
        self.an5 = AdaGN(128, 128, 16, ctx_channels)

        self.up3 = FeaturePropagation(64, 128, [256, 128, 128], use_bn=use_bn)  # b x 3 x 2048
        self.an6 = AdaGN(128, 128, 16, ctx_channels)

        self.final = nn.Sequential(
            nn.Conv1d(128 + 3, 256, 1),
            nn.BatchNorm1d(256) if use_bn else nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 3, 1)
        )

    def forward(self, x, z, time_emb):
        time_emb = self.time_embed(time_emb)
        ctx = torch.cat((z, time_emb), dim=1)
        feat0 = self.an0(x, ctx)

        xyz1, feat1 = self.down1(x, point_features=feat0)
        feat1 = self.an1(feat1, ctx)

        xyz2, feat2 = self.down2(xyz1, point_features=feat1)
        feat2 = self.an2(feat2, ctx)

        xyz3, feat3 = self.down3(xyz2, point_features=feat2)
        feat3 = self.an3(feat3, ctx)
        # feat3 = self.attn1(feat3, feat3)

        feat2 = self.up1(xyz2, xyz3, feat2, feat3)
        feat2 = self.an4(feat2, ctx)

        feat1 = self.up2(xyz1, xyz2, feat1, feat2)
        feat1 = self.an5(feat1, ctx)

        feat0 = self.up3(x, xyz1, feat0, feat1)
        return self.final(torch.cat((x, feat0), dim=1))


class Discriminator(nn.Module):
    def __init__(self, time_emb_size):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.LeakyReLU(0.2),
            nn.Linear(time_emb_size, time_emb_size)
        )

        self.transform_net = TransformNet(use_bn=False, inplace_activation=False)
        self.conv1 = EdgeConv(6, 64, 20, double_mlp=True, use_bn=False, inplace_activation=False)
        self.conv2 = EdgeConv(64 + 3, 64, 30, double_mlp=True, use_bn=False, inplace_activation=False)
        self.conv3 = EdgeConv(64 + 3, 64, 40, use_bn=False, inplace_activation=False)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(64 * 3 + time_emb_size, 1024, 1, bias=False),
            nn.GroupNorm(64, 1024),
            nn.LeakyReLU(0.2)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024 + 64 * 3 + time_emb_size, 1024, 1, bias=False),
            nn.GroupNorm(64, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(1024, 512, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 1)
        )

        self.logit = nn.Sequential(
            nn.Linear(512 * 2 + time_emb_size, 2048),
            nn.GroupNorm(256, 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, xt, xtp1, time_emb):
        time_emb = self.time_embed(time_emb)

        x1 = self.conv1(torch.cat((xt, xtp1), dim=1))
        x2 = self.conv2(torch.cat((x1, xtp1), dim=1))
        x3 = self.conv3(torch.cat((x2, xtp1), dim=1))

        x = self.mlp1(torch.cat((x1, x2, x3,
                                 time_emb.unsqueeze(2).expand(-1, -1, xt.size(2))), dim=1)).max(dim=2, keepdim=True)[0]
        x = torch.cat((x.expand(-1, -1, x1.size(2)), x1, x2, x3,
                       time_emb.unsqueeze(2).expand(-1, -1, xt.size(2))), dim=1)
        features = self.mlp2(x)
        x = torch.cat((features.max(dim=2)[0], features.mean(dim=2), time_emb), dim=1)

        return self.logit(x)

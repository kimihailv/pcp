import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from ..functional import ball_query


class MLP(nn.Module):
    def __init__(self, in_channels, hid_channels, use_bn=True):
        super().__init__()
        self.mlp = []

        for i, channels in enumerate(hid_channels, start=1):
            self.mlp.append(nn.Conv1d(in_channels, channels, 1))

            if i < len(hid_channels):
                self.mlp.append(nn.BatchNorm1d(channels) if use_bn else nn.GroupNorm(16, channels))
                self.mlp.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            in_channels = channels

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


class AttentionMix(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()

        self.q = nn.Conv1d(in_channels, in_channels * num_heads, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels * num_heads, 1, bias=False)
        self.k = nn.Conv2d(in_channels, in_channels * num_heads, 1, bias=False)
        self.out = nn.Conv1d(in_channels * num_heads, in_channels, 1, bias=False)
        self.num_heads = num_heads
        self.norm_const = in_channels ** .5

    def forward(self, x, y, mask=None):
        # x: b x c x k
        # y: b x c x k x n

        bs = x.size(0)
        x_pts = x.size(2)
        total_pts = y.size(3)

        q = self.q(x).view(bs, self.num_heads, -1, x_pts)  # b x h x c x k
        k = self.k(y).view(bs, self.num_heads, -1, x_pts, total_pts)  # b x h x c x k x n
        v = self.v(y).view(bs, self.num_heads, -1, x_pts, total_pts)  # b x h x c x k x n

        w = torch.einsum('bhck, bhckn -> bhkn', q, k) / self.norm_const  # b x h x k x n
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(1), -1e9)

        w = F.softmax(w, dim=3)
        out = torch.einsum('bhkn, bhckn -> bhck', w, v).contiguous().view(bs, -1, x_pts)  # b x h x c x k

        return self.out(out) + x


class Attention1d(nn.Module):
    def __init__(self, q_channels, k_channels, out_channels, num_heads=2):
        super().__init__()
        self.q = nn.Conv1d(q_channels, out_channels * num_heads, 1, bias=False)
        self.v = nn.Conv1d(k_channels, out_channels * num_heads, 1, bias=False)
        self.k = nn.Conv1d(k_channels, out_channels * num_heads, 1, bias=False)
        self.out = nn.Conv1d(out_channels * num_heads, out_channels, 1, bias=False)
        self.num_heads = num_heads
        self.norm_const = out_channels**.5

    def forward(self, x, y):
        bs = x.size(0)
        x_pts = x.size(2)
        y_pts = y.size(2)
        # q: bs x n_heads x Nx x hid_size
        q = self.q(x).view(bs, self.num_heads, -1, x_pts).transpose(-1, -2)
        # k: bs x n_heads x hid_size x Ny
        k = self.k(y).view(bs, self.num_heads, -1, y_pts)
        # v: bs x n_heads x Ny x hid_size
        v = self.v(y).view(bs, self.num_heads, -1, y_pts).transpose(-1, -2)
        # bs x n_heads x Nx x Ny
        logits = torch.einsum('bnik, bnkj->bnij', q, k) / self.norm_const
        attention_probs = F.softmax(logits, dim=-1)
        # out: bs x n_heads * hid_size x Nx
        out = torch.einsum('bnik, bnkj->bnji', attention_probs, v).contiguous().view(bs, -1, x_pts)
        return self.out(out)


class PointNetMSG(nn.Module):
    def __init__(self, n_centroids, max_n_points, radius, in_channels, hid_channels, use_bn=True):
        super().__init__()
        self.n_centroids = n_centroids
        self.max_n_points = max_n_points
        self.radius = radius
        self.mlps = nn.ModuleList()
        self.attention = nn.ModuleList()

        for channels in hid_channels:
            self.attention.append(AttentionMix(in_channels))
            self.mlps.append(MLP(in_channels, channels, use_bn=use_bn))

    def forward(self, xyz, point_features=None):
        """
        :param xyz: point cloud coordinates, b x 3 x n
        :param point_features: pointwise features, b x c x n
        :return: sample of xyz, new features
        """
        features_list = []
        support = xyz if point_features is None else torch.cat([xyz, point_features], dim=1)

        centroids_idx = self.sample(support)  # b x n_centroids
        centroids = torch.gather(support, 2, centroids_idx.unsqueeze(1).expand(-1, support.size(1), -1))
        new_xyz = torch.gather(xyz, 2, centroids_idx.unsqueeze(1).expand(-1, xyz.size(1), -1))
        ex_support = support.unsqueeze(2).expand(-1, -1, self.n_centroids, -1)  # b x (c+3) x n_centroids x n
        for radius, k, mlp, attn in zip(self.radius, self.max_n_points, self.mlps, self.attention):
            # group_idx, mask = self.group(support, centroids, radius, k)
            group_idx = ball_query(centroids, ex_support, radius, k).long()
            # b x c x n_centroids x n_points
            group = torch.gather(ex_support, 3, group_idx.unsqueeze(1).expand(-1, support.size(1), -1, -1))
            group -= centroids.unsqueeze(3)

            features = attn(centroids, group)
            features = mlp(features)
            features_list.append(features)

        features = torch.cat(features_list, dim=1)

        return new_xyz, features

    def sample(self, x):
        device = x.device
        B, C, N = x.shape
        centroids = torch.zeros(B, self.n_centroids, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        for i in range(self.n_centroids):
            centroids[:, i] = farthest
            centroid = x[batch_indices, :, farthest].view(B, C, 1)
            dist = torch.sum((x - centroid) ** 2, 1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

    def group(self, x, centroids, radius, k):
        # x: b x c x n
        # centroids: b x c x n_centroids

        batch_size, n_points = x.size(0), x.size(2)
        n_centroids = centroids.size(2)

        dists = (
                x.pow(2).sum(dim=1, keepdim=True) -
                2 * torch.bmm(centroids.transpose(2, 1), x)
                + centroids.pow(2).sum(dim=1).unsqueeze(2)
        )  # b x m x n

        idx = torch.arange(n_points, device=x.device).view(1, 1, n_points).expand(batch_size, n_centroids, -1).clone()
        idx[dists > radius**2] = n_points
        idx = torch.topk(idx, k, dim=2, largest=False)[0]  # deterministic neighbourhood size restriction
        first_point_idx = idx[:, :, 0:1].expand(batch_size, n_centroids, k).clone()
        first_point_idx.masked_fill_(first_point_idx == n_points, 0)
        mask = idx == n_points
        idx[mask] = first_point_idx[mask]

        # b x n_centroids x k
        return idx, mask


class AdaGN(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, latent_channels,
                 use_linear=True):
        super().__init__()

        self.linear = nn.Conv1d(in_channels, out_channels, 1) if use_linear else nn.Identity()
        self.norm = nn.GroupNorm(num_groups, in_channels, affine=False, eps=1e-6)
        self.scale_shift = nn.Linear(latent_channels, out_channels * 2)
        self.scale_shift.bias.data[:out_channels] = 1
        self.scale_shift.bias.data[out_channels:] = 0

    def forward(self, x, z):
        x = self.norm(self.linear(x))
        scale, shift = self.scale_shift(z).unsqueeze(2).chunk(2, 1)
        return x * scale + shift


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, n_centroids):
        super().__init__()
        self.mlp = MLP(in_channels, hid_channels)
        self.n_centroids = n_centroids
        self.attention = Attention1d(hid_channels[-1], hid_channels[-1], hid_channels[-1], num_heads=1)

    def forward(self, xyz, point_features=None):
        support = xyz if point_features is None else torch.cat([xyz, point_features], dim=1)
        centroids_idx = self.sample(support)
        features = self.mlp(support)
        centroids = torch.gather(features, 2, centroids_idx.unsqueeze(1).expand(-1, features.size(1), -1))
        new_xyz = torch.gather(xyz, 2, centroids_idx.unsqueeze(1).expand(-1, xyz.size(1), -1))
        return new_xyz, centroids + self.attention(centroids, features)

    def sample(self, x):
        device = x.device
        B, C, N = x.shape
        centroids = torch.zeros(B, self.n_centroids, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        for i in range(self.n_centroids):
            centroids[:, i] = farthest
            centroid = x[batch_indices, :, farthest].view(B, C, 1)
            dist = torch.sum((x - centroid) ** 2, 1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids


class FeaturePropagation(nn.Module):
    def __init__(self, x_in_channels, y_in_channels, hid_channels, use_bn=True):
        super().__init__()
        self.layers = []

        self.attention = Attention1d(x_in_channels + 3, y_in_channels + 3, y_in_channels)
        in_channels = x_in_channels + y_in_channels
        self.mlp = MLP(in_channels, hid_channels, use_bn=use_bn)

    def forward(self, x, y, x_features, y_features):
        x = torch.cat((x, x_features), dim=1) if x_features is not None else x
        y = torch.cat((y, y_features), dim=1)

        interpolated = self.attention(x, y)
        if x_features is not None:
            interpolated = torch.cat((interpolated, x_features), dim=1)

        return self.mlp(interpolated)


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k, use_bn=True, double_mlp=False, inplace_activation=True):
        super().__init__()
        self.k = k
        norm = nn.BatchNorm2d(out_channels) if use_bn else nn.GroupNorm(16, out_channels)

        if double_mlp:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=inplace_activation),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=inplace_activation)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                norm,
                nn.LeakyReLU(0.2, inplace=inplace_activation),
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
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

    def forward(self, x):
        x = self.get_graph_feature(x)
        return self.conv(x).max(dim=-1)[0]


class InvertibleConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        rotation_mat = torch.linalg.qr(torch.randn(in_channels, in_channels))[0]
        perm, lower, upper = torch.lu_unpack(*rotation_mat.lu())
        s = upper.diag()
        upper = upper.triu(1)
        logs = s.abs().log()
        s_sign = s.sign()
        mask = torch.triu(torch.ones(in_channels, in_channels), diagonal=1)
        identity = torch.eye(in_channels)
        self.register_buffer('perm', perm)
        self.register_buffer('mask', mask)
        self.register_buffer('identity', identity)
        self.register_buffer('s_sign', s_sign)

        self.register_parameter('lower', nn.Parameter(lower))
        self.register_parameter('upper', nn.Parameter(upper))
        self.register_parameter('logs', nn.Parameter(logs))

    def get_weight(self, reverse):
        lower = self.lower * self.mask.t() + self.identity
        upper = self.upper * self.mask + torch.diag(self.logs.exp() * self.s_sign)
        logdet = self.logs.sum()

        if not reverse:
            weight = self.perm @ lower @ upper
        else:
            perm_inv = self.perm.inverse()
            lower_inv = lower.inverse()
            upper_inv = upper.inverse()
            logdet = -logdet
            weight = upper_inv @ lower_inv @ perm_inv

        return weight, logdet

    def forward(self, x, reverse):
        w, log_det = self.get_weight(reverse)
        return x @ w, log_det


class ActNorm(nn.Module):
    def __init__(self, in_channels, return_log_det=True):
        super().__init__()
        self.register_parameter('shift', nn.Parameter(torch.zeros(1, in_channels)))
        self.register_parameter('scale', nn.Parameter(torch.zeros(1, in_channels)))
        self.register_buffer('inited', torch.zeros(1))
        self.return_log_det = return_log_det

    @torch.no_grad()
    def initialize(self, x):
        self.shift.data.copy_(-x.mean(dim=0, keepdim=True))
        std = x.std(dim=0, keepdim=True) + 1e-6
        self.scale.data.copy_(1 / std)
        self.inited += 1

    def forward(self, x, reverse=False):
        if self.inited[0].item() == 0:
            self.initialize(x)

        log_det = self.scale.abs().log().sum()
        if not reverse:
            x = (x + self.shift) * self.scale
        else:
            x = x / self.scale - self.shift
            log_det = -log_det

        if self.return_log_det:
            return x, log_det

        return x


class ZeroLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.register_parameter('scale', nn.Parameter(torch.zeros(1, out_channels)))

    def forward(self, x):
        out = self.linear(x)
        return out * (self.scale * 3).exp()


class AffineCoupling(nn.Module):
    def __init__(self, in_channels, mid_channels, use_actnorm=False):
        super().__init__()
        self.s_t = nn.Sequential(
            nn.Linear(in_channels // 2, mid_channels),
            ActNorm(mid_channels, return_log_det=False) if use_actnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels, 1),
            ActNorm(mid_channels, return_log_det=False) if use_actnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            ZeroLinear(mid_channels, in_channels)
        )

        self.s_t[0].weight.data.normal_(0, 0.05)
        self.s_t[0].bias.data.zero_()

        self.s_t[3].weight.data.normal_(0, 0.05)
        self.s_t[3].bias.data.zero_()

    def forward(self, x, reverse):
        x_a, x_b = x.chunk(2, 1)
        log_s, t = self.s_t(x_a).chunk(2, 1)

        s = torch.sigmoid(log_s + 2)
        log_det = s.log().sum(dim=1)

        if not reverse:
            x_b = (x_b + t) * s
            x = torch.cat((x_a, x_b), dim=1)
        else:
            x_b = x_b / s - t
            log_det = -log_det
            x = torch.cat((x_a, x_b), dim=1)
        return x, log_det


class FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.act_norm = ActNorm(in_channels)
        self.inv_conv = InvertibleConv(in_channels)
        self.affine_coupling = AffineCoupling(in_channels, mid_channels)

    def forward(self, x, reverse):
        if not reverse:
            z, log_det = self.act_norm(x, reverse)
            z, log_det1 = self.inv_conv(z, reverse)
            z, log_det2 = self.affine_coupling(z, reverse)
            return z, log_det + log_det1 + log_det2

        z, log_det = self.affine_coupling(x, reverse)
        z, log_det1 = self.inv_conv(z, reverse)
        z, log_det2 = self.act_norm(z, reverse)
        return z, log_det + log_det1 + log_det2


class TransformNet(nn.Module):
    def __init__(self, use_bn=True, inplace_activation=True):
        super().__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.GroupNorm(8, 64)
        self.bn2 = nn.BatchNorm2d(128) if use_bn else nn.GroupNorm(16, 128)
        self.bn3 = nn.BatchNorm1d(1024) if use_bn else nn.GroupNorm(64, 1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=inplace_activation)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=inplace_activation)
                                   )
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2, inplace=inplace_activation)
                                   )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512) if use_bn else nn.GroupNorm(32, 512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256) if use_bn else nn.GroupNorm(16, 256)

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

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


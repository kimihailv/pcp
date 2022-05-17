import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm


class Attention1d(nn.Module):
    def __init__(self, q_channels, k_channels, out_channels, num_heads=1):
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


class MiniPointNet(nn.Module):
    n_output = 512

    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(512 + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
        )

        self.attn = Attention1d(512, 512, 512)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.n_output)
        )

    def forward(self, x):
        global_emb = self.mlp1(x).max(dim=2, keepdim=True)[0].expand(-1, -1, x.size(2))
        x = torch.cat((global_emb, x), dim=1)
        x = self.mlp2(x)
        x = self.attn(x, x).max(dim=2)[0]
        return self.head(x), 0

    def forward_features(self, x):
        global_emb = self.mlp1(x).max(dim=2, keepdim=True)[0].expand(-1, -1, x.size(2))
        x = torch.cat((global_emb, x), dim=1)
        x = self.mlp2(x)
        # x = self.attn(x, x)
        return x


class AE(nn.Module):
    def __init__(self, time_size):
        super().__init__()
        self.encoder = MiniPointNet()
        code_size = self.encoder.n_output

        self.time_embed = nn.Sequential(
            nn.Linear(time_size, time_size * 2),
            nn.BatchNorm1d(time_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(time_size * 2, time_size)
        )
        ctx_size = code_size + time_size

        self.an1 = AdaGN(3, 128, 8, ctx_size)
        self.an2 = AdaGN(128, 128, 8, ctx_size)

        self.an3 = AdaGN(128, 256, 32, ctx_size)
        self.an4 = AdaGN(256, 256, 32, ctx_size)

        self.an5 = AdaGN(256, 512, 32, ctx_size)
        self.an6 = AdaGN(512, 512, 32, ctx_size)

        self.linear = nn.Conv1d(3, 3, 1)
        self.linear.weight.data = torch.eye(3).unsqueeze(2)
        self.linear.bias.data.zero_()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.mlp = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(1024, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 3, 1)
        )

    def encode(self, x):
        z, reg = self.encoder(x)
        return z, reg

    def decode(self, xt, z, time_emb, return_features):
        ctx_emb = torch.cat([self.time_embed(time_emb), z], dim=1)
        f1 = self.act(self.an1(xt, ctx_emb))
        f1 = self.act(self.an2(f1, ctx_emb) + f1)

        f2 = self.act(self.an3(f1, ctx_emb))
        f2 = self.act(self.an4(f2, ctx_emb) + f2)

        f3 = self.act(self.an5(f2, ctx_emb))
        f3 = self.act(self.an6(f3, ctx_emb) + f3)

        x = self.mlp(f3) + self.linear(xt)

        if return_features:
            features = [f1, f2, f3]
            return x, features

        return x

    def forward(self, x, xt, time_emb, return_features=False, return_latent=False):
        z, reg = self.encode(x)
        et = self.decode(xt, z, time_emb, return_features)

        if return_latent:
            return et, z, reg

        return et, reg


class DiffAE(nn.Module):
    def __init__(self, total_steps=150, time_size=128):
        super().__init__()
        self.total_steps = total_steps
        self.time_size = time_size
        self.ae = AE(time_size)

        alpha = torch.arange(self.total_steps, dtype=torch.double) / self.total_steps
        alpha = (alpha + 0.008) / 1.008 * math.pi / 2
        alpha = alpha.cos().pow(2)
        # alpha = 1 - torch.linspace(0.0001, 0.05, total_steps, dtype=torch.float32)
        self.register_buffer('alpha', alpha.float())

    def sample_xt(self, x0, timesteps):
        a = torch.index_select(self.alpha, 0, timesteps).view(-1, 1, 1)
        e = torch.randn_like(x0)
        return x0 * a.sqrt() + e * (1 - a).sqrt(), e

    def forward(self, x, return_latent=False, return_features=False):
        batch_size = x.size(0)
        timesteps = torch.randint(
            low=0, high=self.total_steps, size=(batch_size // 2 + 1,)
        ).to(self.alpha.device)
        timesteps = torch.cat([timesteps, self.total_steps - timesteps - 1], dim=0)[:batch_size]
        time_emb = self.get_time_embeddings(timesteps)
        xt, e = self.sample_xt(x, timesteps)
        out = self.ae(x, xt, time_emb, return_latent=return_latent, return_features=return_features)
        return e, out

    def forward_with_latent(self, xt, z, time_emb):
        return self.ae.decode(xt, z, time_emb, return_features=False)

    def get_time_embeddings(self, timesteps):
        half_dim = self.time_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.time_size % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    @torch.no_grad()
    def deterministic_forward_process(self, x, use_tqdm, steps=-1):
        steps = steps if steps > 0 else self.total_steps
        bar = range(1, steps)

        if use_tqdm:
            bar = tqdm(bar)

        z, _ = self.ae.encode(x)

        for t in bar:
            alpha_tm1 = self.alpha[t - 1]
            alpha_t = self.alpha[t]
            timesteps = torch.ones(x.size(0), device=self.alpha.device, dtype=torch.int64) * (t - 1)
            time_emb = self.get_time_embeddings(timesteps)
            et = self.ae.decode(x, z, time_emb, return_features=False)
            x0_pred = (x - et * (1 - alpha_tm1).sqrt()) / alpha_tm1.sqrt()
            x = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * et

        return x

    @torch.no_grad()
    def _reconstruct(self, x, z, use_tqdm, steps=-1):
        steps = steps if steps > 0 else self.total_steps
        bar = range(steps - 1, 0, -1)
        if use_tqdm:
            bar = tqdm(bar)

        for t in bar:
            alpha_t = self.alpha[t]
            alpha_prev_t = self.alpha[t - 1]
            timesteps = torch.ones(x.size(0), device=x.device, dtype=torch.int64) * t
            time_emb = self.get_time_embeddings(timesteps)

            et = self.ae.decode(x, z, time_emb, return_features=False)
            x0_t = (x - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
            eps = (x / alpha_t.sqrt() - x0_t) / (1 / alpha_t - 1).sqrt()
            x = alpha_prev_t.sqrt() * x0_t + (1 - alpha_prev_t).sqrt() * eps

        return x

    def interpolate(self, x1, x2, alpha, deterministic=True, steps=-1):
        x = torch.cat((x1, x2), dim=0)
        z, _ = self.ae.encode(x)
        z1, z2 = z.chunk(2, 0)

        if deterministic:
            x1, x2 = self.deterministic_forward_process(x, use_tqdm=False, steps=steps).chunk(2, 0)
            x = self.slerp(x1, x2, alpha)
        else:
            x = torch.randn(x.size(0) // 2, 3, 2048, device=x.device)

        z = z1 * alpha + (1 - alpha) * z2

        return self._reconstruct(x, z, use_tqdm=False, steps=steps)

    def slerp(self, p1, p2, alpha):
        p1_n = p1.view(p1.size(0), -1).contiguous()
        p2_n = p2.view(p2.size(0), -1).contiguous()

        p1_n = F.normalize(p1_n, dim=1)
        p2_n = F.normalize(p2_n, dim=1)

        cos = (p1_n * p2_n).sum(dim=1)
        theta = cos.acos()
        sin = theta.sin()
        c1 = (theta * (1 - alpha)).sin() / sin
        c2 = (theta * alpha).sin() / sin

        return c1[:, None, None] * p1 + c2[:, None, None] * p2

    def auto_encode(self, x, deterministic=True, use_tqdm=False):
        z, _ = self.ae.encode(x)
        if deterministic:
            x = self.deterministic_forward_process(x, use_tqdm)
        else:
            x = torch.randn_like(x)

        return self._reconstruct(x, z, use_tqdm)

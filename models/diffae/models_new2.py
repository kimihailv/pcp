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
            nn.Linear(256, in_channels ** 2)
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
        return coef * (t - identity).pow(2).sum().div(2)


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


class DownSample(nn.Module):
    def __init__(self, input_size, output_size, n_centroids):
        super(DownSample, self).__init__()
        self.attention = Attention1d(input_size + 3, input_size + 3, output_size)
        self.mlp = nn.Sequential(
            nn.Conv1d(output_size, output_size, 1),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(output_size, output_size, 1)
        )
        self.n_centroids = n_centroids

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

    def forward(self, x, xt, idx=None):
        if idx is None:
            x = torch.cat((x, xt), dim=1)
        else:
            xt = torch.gather(xt, 2, idx.unsqueeze(1).expand(-1, 3, -1))
            x = torch.cat((x, xt), dim=1)

        centroids_idx = self.sample(x)

        centroids = torch.gather(x, 2, centroids_idx.unsqueeze(1).expand(-1, x.size(1), -1))
        centroids = self.attention(centroids, x)
        return self.mlp(centroids), centroids_idx


class UpSample(nn.Module):
    def __init__(self, query_size, key_size, output_size):
        super().__init__()
        self.attention = Attention1d(query_size, key_size, key_size)
        self.mlp = nn.Sequential(
            nn.Conv1d(query_size + key_size, output_size, 1),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(output_size, output_size, 1)
        )

    def forward(self, q, k):
        upsampled = self.attention(q, k)
        upsampled = torch.cat((upsampled, q), dim=1)
        return self.mlp(upsampled)


class PatchEmbedder(nn.Module):
    def __init__(self, n_embs, emb_dim, input_dim, total_steps, t_min=0.05, t_max=2/3, kl_weight=5e-4):
        super().__init__()
        self.n_embs = n_embs
        self.register_parameter('vocab', nn.Parameter(torch.randn(n_embs, emb_dim)))
        self.t_min = t_min
        self.t_max = t_max
        self.embedder = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim, n_embs, 1)
        )
        self.kl_weight = kl_weight
        self.total_steps = total_steps
        self.current_step = 0

    def forward(self, x, hard=False):
        logits = self.embedder(x)  # b x n_embs x n_points
        z = F.gumbel_softmax(logits, tau=self.get_current_tau(self.current_step), hard=hard, dim=1)
        z_q = torch.einsum('bnk,nd -> bdk', z, self.vocab)
        ind = z.argmax(dim=1)
        p = F.softmax(logits, dim=1)
        kl = self.kl_weight * torch.sum(p * torch.log(p * self.n_embs + 1e-10), dim=1).mean()
        if self.training:
            self.current_step += 1
        return z_q, ind, -kl

    def get_current_tau(self, step):
        tau = self.t_min + 0.5 * (self.t_max - self.t_min) * (1 + math.cos(step / self.total_steps * math.pi))
        return tau


class MiniPointNet(nn.Module):
    n_output = 512

    def __init__(self):
        super().__init__()
        self.tn3 = TransformNet(3)

        self.mlp11 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1)
        )

        self.tn64 = TransformNet(64, True)
        self.attn1 = Attention1d(64, 64, 128, 2)

        self.mlp12 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1)
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

        self.attn2 = Attention1d(512, 512, 512)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.n_output)
        )

    def forward(self, x):
        x = self.tn3(x)
        x1 = self.mlp11(x)
        x1, reg = self.tn64(x1)
        x1 = self.attn1(x1, x1)
        global_emb = self.mlp12(x1).max(dim=2, keepdim=True)[0].expand(-1, -1, x.size(2))
        x = torch.cat((global_emb, x), dim=1)
        x = self.mlp2(x)
        x = self.attn2(x, x).max(dim=2)[0]
        return self.head(x), reg


class AE(nn.Module):
    def __init__(self, time_size, total_steps):
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
        # self.attn1 = Attention1d(128, 128, 128)

        self.an3 = AdaGN(128, 256, 32, ctx_size)
        self.an4 = AdaGN(256, 256, 32, ctx_size)

        self.an5 = AdaGN(256, 512, 32, ctx_size)
        self.an6 = AdaGN(512, 512, 32, ctx_size)

        self.linear = nn.Conv1d(3, 3, 1)
        self.linear.weight.data = torch.eye(3).unsqueeze(2)
        self.linear.bias.data.zero_()
        self.act = nn.SiLU(inplace=True)  # nn.LeakyReLU(0.2, inplace=True)

        self.patch_embedder = PatchEmbedder(n_embs=4,
                                            emb_dim=256,
                                            input_dim=512,
                                            total_steps=total_steps)
        self.mlp = nn.Sequential(
            nn.Conv1d(512 + 256, 1024, 1),
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
        f1 = self.an1(xt, ctx_emb)
        f1 = self.act(self.an2(f1, ctx_emb) + f1)
        # f1 = self.attn1(f1, f1)

        f2 = self.an3(f1, ctx_emb)
        f2 = self.act(self.an4(f2, ctx_emb) + f2)

        f3 = self.an5(f2, ctx_emb)
        f3 = self.act(self.an6(f3, ctx_emb) + f3)

        patch_embs, ind, kl = self.patch_embedder(f3)

        x = self.mlp(torch.cat((f3, patch_embs), dim=1)) + self.linear(xt)

        if return_features:
            features = (f1, f2, f3)
            return x, features, ind

        return x, kl

    def forward(self, x, xt, time_emb, return_features=False, return_latent=False):
        z, reg = self.encode(x)
        et = self.decode(xt, z, time_emb, return_features)

        kl = 0
        if not return_features:
            et, kl = et

        if return_latent:
            return et, z, reg + kl

        return et, reg + kl


class DiffAE(nn.Module):
    def __init__(self, train_steps, total_steps=1000, time_size=128):
        super().__init__()
        self.total_steps = total_steps
        self.time_size = time_size
        self.ae = AE(time_size, total_steps=train_steps)

        alpha = 1 - torch.linspace(0.0001, 0.05, total_steps, dtype=torch.float32)
        self.register_buffer('alpha', alpha.cumprod(dim=0).float())

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
            if isinstance(et, tuple):
                et = et[0]
            x0_pred = (x - et * (1 - alpha_tm1).sqrt()) / alpha_tm1.sqrt()
            x = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * et

        return x

    @torch.no_grad()
    def _reconstruct(self, x, z, use_tqdm, steps=-1):
        steps = steps if steps > 0 else self.total_steps
        seq = range(1, self.total_steps, self.total_steps // steps)
        seq_prev = [0] + list(seq)[:-1]
        bar = zip(reversed(seq_prev), reversed(seq))

        if use_tqdm:
            bar = tqdm(bar, total=len(seq_prev))

        for prev_t, t in bar:
            alpha_t = self.alpha[t]
            alpha_prev_t = self.alpha[prev_t]
            timesteps = torch.ones(x.size(0), device=x.device, dtype=torch.int64) * t
            time_emb = self.get_time_embeddings(timesteps)

            et = self.ae.decode(x, z, time_emb, return_features=False)
            if isinstance(et, tuple):
                et = et[0]
            x0_t = (x - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
            eps = (x / alpha_t.sqrt() - x0_t) / (1 / alpha_t - 1).sqrt()
            x = alpha_prev_t.sqrt() * x0_t + (1 - alpha_prev_t).sqrt() * eps

        return x

    def interpolate(self, x1, x2, alpha, deterministic=True, use_tqdm=False, steps=-1):
        x = torch.cat((x1, x2), dim=0)
        z, _ = self.ae.encode(x)
        z1, z2 = z.chunk(2, 0)

        if deterministic:
            x1, x2 = self.deterministic_forward_process(x, use_tqdm=use_tqdm, steps=steps).chunk(2, 0)
            x = self.slerp(x1, x2, alpha)
        else:
            x = torch.randn(x.size(0) // 2, 3, 2048, device=x.device)

        z = z1 * alpha + (1 - alpha) * z2

        return self._reconstruct(x, z, use_tqdm=use_tqdm, steps=steps)

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

    def auto_encode(self, x, deterministic=True, use_tqdm=False, steps=-1):
        z, _ = self.ae.encode(x)
        if deterministic:
            x = self.deterministic_forward_process(x, use_tqdm=use_tqdm, steps=steps)
        else:
            x = torch.randn_like(x)

        return self._reconstruct(x, z, use_tqdm=use_tqdm, steps=steps)

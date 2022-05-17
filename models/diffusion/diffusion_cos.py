import torch
import torch.nn as nn
import math
from tqdm import tqdm


class DiffusionModel(nn.Module):
    def __init__(self, extractor, total_steps, time_embedding_dim=64, eta=0):
        super().__init__()
        self.extractor = extractor
        self.total_steps = total_steps
        self.time_embedding_dim = time_embedding_dim
        self.eta = eta
        alpha_bar, beta = self.get_schedules()
        '''beta = torch.linspace(
            0.0001, 0.05, total_steps, dtype=torch.float32
        )
        beta = torch.cat((torch.zeros(1), beta), dim=0)'''
        alpha = 1 - beta
        # alpha_bar = alpha.cumprod(dim=0)
        var = torch.zeros_like(beta)

        for i in range(1, total_steps + 1):
            var[i] = ((1 - alpha_bar[i - 1]) / (1 - alpha_bar[i])) * beta[i]

        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('beta', beta)
        self.register_buffer('sigma', var.sqrt())

    def get_schedules(self):
        def compute_alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        beta = [0]
        alpha_bar = [0]
        for i in range(self.total_steps):
            t1 = i / self.total_steps
            t2 = (i + 1) / self.total_steps
            alpha_bar1 = compute_alpha_bar(t1)
            beta.append(min(1 - compute_alpha_bar(t2) / alpha_bar1, 0.999))
            alpha_bar.append(alpha_bar1)

        alpha_bar = torch.FloatTensor(alpha_bar)
        beta = torch.FloatTensor(beta)

        return alpha_bar, beta

    def forward(self, x):
        batch_size = x.size(0)
        # timesteps = torch.randint(low=1, high=self.total_steps + 1, size=(batch_size,), device=x.device)
        timesteps = torch.randint(
            low=1, high=self.total_steps + 1, size=(batch_size // 2 + 1,)
        ).to(self.alpha.device)
        timesteps = torch.cat([timesteps, self.total_steps - timesteps + 1], dim=0)[:batch_size]
        time_emb = self.get_time_embeddings(self.time_embedding_dim, timesteps)
        xt, e = self.sample_xt(x, timesteps)
        out = self.extractor(x, xt, time_emb, return_features=False)
        return e, out

    def sample_xt(self, x0, timesteps):
        a = torch.index_select(self.alpha_bar, 0, timesteps).view(-1, 1, 1)
        e = torch.randn_like(x0)
        return x0 * a.sqrt() + e * (1 - a).sqrt(), e

    @torch.no_grad()
    def sample(self, x_current=None, x0=None, batch_size=8,
               track_trajectory=False, skip=1):

        if x0 is not None:
            x_current, _ = self.sample_xt(x0, torch.ones(x0.size(0),
                                                         device=self.alpha.device,
                                                         dtype=torch.int64) * self.total_steps)
        elif x_current is None:
            x_current = torch.randn(batch_size, 3, 2048, device=self.alpha.device)

        if track_trajectory:
            trajectory = [x_current.detach().cpu()]

        latent = None
        seq = range(1, self.total_steps + 1, skip)
        seq_prev = [0] + list(seq)[:-1]
        for prev_t, t in tqdm(zip(reversed(seq_prev), reversed(seq)), total=len(seq_prev)):
            if track_trajectory:
                trajectory.append(x_current.cpu())

            alpha_t = self.alpha_bar[t]
            alpha_prev_t = self.alpha_bar[prev_t]
            timesteps = torch.ones(batch_size, device=self.alpha.device, dtype=torch.int64) * t
            time_emb = self.get_time_embeddings(self.time_embedding_dim, timesteps)

            et, (latent, _, _) = self.extractor(x0, x_current, time_emb,
                                                return_features=False, z=latent,
                                                deterministic_latent=True)

            x0_t = (x_current - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
            c1 = (
                    self.eta * ((1 - alpha_t / alpha_prev_t) * (1 - alpha_prev_t) / (1 - alpha_t)).sqrt()
            )
            c2 = (1 - alpha_prev_t - c1 ** 2).sqrt()
            x_current = alpha_prev_t.sqrt() * x0_t + c1 * torch.randn_like(x_current) + c2 * et

        if track_trajectory:
            return x_current, trajectory

        return x_current

    @torch.no_grad()
    def sample_ddpm(self, x_current=None, x0=None, batch_size=8,
                    track_trajectory=False, return_features=False):
        batch_size = batch_size if x_current is None else x_current.size(0)

        if x0 is not None:
            x_current, _ = self.sample_xt(x0, torch.ones(x0.size(0),
                                                         device=self.alpha.device,
                                                         dtype=torch.int64) * self.total_steps)
        elif x_current is None:
            x_current = torch.randn(batch_size, 3, 2048, device=self.alpha.device)

        if track_trajectory:
            trajectory = [x_current.detach().cpu()]

        latent = None

        for timestep in tqdm(range(self.total_steps, 0, -1)):
            if track_trajectory:
                trajectory.append(x_current.detach().cpu())

            timesteps = torch.IntTensor([timestep]).to(self.alpha.device).expand(batch_size)
            time_emb = self.get_time_embeddings(self.time_embedding_dim, timesteps)

            if not return_features:
                et, (latent, _, _) = self.extractor(x0, x_current, time_emb, return_features=False, z=latent,
                                                    deterministic_latent=True)
            else:
                (features, _), (et, latent, _, _) = self.extractor(x0, x_current, time_emb,
                                                                   return_features=True, z=latent,
                                                                   deterministic_latent=True)

            c = (1 - self.alpha[timestep]) / (1 - self.alpha_bar[timestep]) ** 0.5
            z = 0 if timestep == 1 else torch.randn_like(x_current, device=self.alpha.device)
            x_current = (x_current - c * et) / self.alpha[timestep] ** 0.5 + self.sigma[timestep] * z

        if not return_features:
            if track_trajectory:
                return x_current, trajectory

            return x_current

        if track_trajectory:
            return x_current, trajectory, features

        return x_current, features

    @torch.no_grad()
    def get_features(self, x, timesteps):
        latent = None

        features_list = {}
        # coords_list = {}
        for timestep in timesteps:
            t = torch.IntTensor([timestep]).to(self.alpha.device).expand(x.size(0))
            xt, _ = self.sample_xt(x, t)
            time_emb = self.get_time_embeddings(self.time_embedding_dim, t)
            (features, coords), (et, z, _, _) = self.extractor(x, xt, time_emb,
                                                               return_features=True,
                                                               z=latent,
                                                               deterministic_latent=True)
            features_list[timestep] = features
            # coords_list[timestep] = coords

        return features_list  # ,coords_list

    def get_time_embeddings(self, dim, timesteps):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        # beta = torch.index_select(self.beta, 0, timesteps).unsqueeze(1)
        # embedding = torch.cat([beta, beta.sin(), beta.cos()], dim=1)
        return emb


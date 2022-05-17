import torch
import torch.nn as nn
from math import pi, log
from .modules import EdgeConv, FlowStep, ZeroLinear, TransformNet


class Encoder1(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.transform_net = TransformNet(use_bn=True)
        self.conv1 = EdgeConv(3, 64, 20, double_mlp=True, use_bn=True)
        self.conv2 = EdgeConv(64, 64, 30, double_mlp=True, use_bn=True)
        self.conv3 = EdgeConv(64, 64, 40, use_bn=True)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(64 * 3, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(512 + 64 * 3, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(1024, latent_size, 1, bias=False),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(latent_size, latent_size, 1)
        )

        self.mean = nn.Sequential(
            nn.Linear(latent_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, latent_size)
        )

        self.log_var = nn.Sequential(
            nn.Linear(latent_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, latent_size)
        )

        self.mean[-1].weight.data.zero_()
        self.mean[-1].bias.data.zero_()

        self.log_var[-1].weight.data.zero_()
        self.log_var[-1].bias.data.zero_()

        self.flow = Flow(latent_size)

    def forward_features(self, x):
        x0 = self.conv1.get_graph_feature(x)
        t = self.transform_net(x0)
        x = torch.bmm(x.transpose(2, 1), t)
        x = x.transpose(2, 1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x = self.mlp1(torch.cat((x1, x2, x3), dim=1)).max(dim=-1, keepdim=True)[0]
        x = torch.cat((x.expand(-1, -1, x1.size(2)), x1, x2, x3), dim=1)
        x = self.mlp2(x)

        return x

    def forward(self, x):
        features = self.forward_features(x)
        x = torch.cat((features.max(dim=2)[0], features.mean(dim=2)), dim=1)
        mu = self.mean(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def reparameterize(self, mu, log_var):
        eps = torch.randn_like(mu)
        return eps * (log_var * 0.5).exp() + mu

    def evaluate_loglikelihood(self, z):
        w, log_det = self.flow(z, False)
        return self.flow.compute_logpw(w) + log_det


class Encoder(nn.Module):
    def __init__(self, latent_size):
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

        self.mean = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_size)
        )

        self.log_var = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_size)
        )

        self.mean[-1].weight.data.zero_()
        self.mean[-1].bias.data.zero_()

        self.log_var[-1].weight.data.zero_()
        self.log_var[-1].bias.data.zero_()

        self.flow = Flow(latent_size)

    def forward(self, x):
        global_emb = self.mlp1(x).max(dim=2, keepdim=True)[0].expand(-1, -1, x.size(2))
        x = torch.cat((global_emb, x), dim=1)
        x = self.mlp2(x).max(dim=2)[0]
        mean = self.mean(x)
        log_var = self.log_var(x)

        return self.reparameterize(mean, log_var), mean, log_var

    def reparameterize(self, mean, log_var):
        std = (log_var * 0.5).exp()
        e = torch.randn_like(std)

        return std * e + mean

    def evaluate_loglikelihood(self, z):
        w, log_det = self.flow(z, False)
        return self.flow.compute_logpw(w) + log_det


class Flow(nn.Module):
    def __init__(self, in_channels, n_blocks=5):
        super().__init__()
        self.in_channels = in_channels
        self.steps = nn.ModuleList([FlowStep(in_channels, 512) for _ in range(n_blocks)])
        self.prior_transform = ZeroLinear(in_channels, in_channels * 2)
        self.register_buffer('prior', torch.zeros(1, in_channels))

    def forward(self, x, reverse):
        log_det_sum = 0
        if not reverse:
            for step in self.steps:
                x, log_det = step(x, reverse)
                log_det_sum += log_det
        else:
            for step in self.steps[::-1]:
                x, log_det = step(x, reverse)
                log_det_sum += log_det

        return x, log_det_sum

    def sample(self, batch_size, temperature=1):
        mean, log_std = self.get_prior()
        eps = torch.randn(batch_size, self.in_channels, device=mean.device)

        w = eps * temperature * log_std.exp() + mean

        return self.forward(w, True)[0]

    def get_prior(self):
        return self.prior_transform(self.prior).chunk(2, 1)

    def compute_logpw(self, w):
        mean, log_std = self.get_prior()
        log_likelihood = -0.5 * (log_std * 2 + (w - mean) ** 2 / (log_std * 2).exp() + log(2 * pi))
        return log_likelihood.sum(dim=1)
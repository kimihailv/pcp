import torch
import torch.nn.functional as F
import torch.nn as nn
from abc import ABC, abstractmethod


class ProjectionHead(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels):
        super().__init__()
        self.layers = []
        for i in range(n_layers):
            if i != n_layers - 1:
                self.layers.append(nn.Linear(in_channels, in_channels))
                self.layers.append(nn.BatchNorm1d(in_channels))
                self.layers.append(nn.ReLU(True))
            else:
                self.layers.append(nn.Linear(in_channels, out_channels, bias=False))
                # self.layers.append(nn.BatchNorm1d(out_channels))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, idx=-1):
        if idx == -1:
            idx = len(self.layers)
        for layer in self.layers[:idx]:
            x = layer(x)

        return x


class BaseModel(nn.Module, ABC):
    n_output_inst = -1
    n_output_point = -1
    projection_dim = 128

    def __init__(self, flatten_embeddings: bool = False):
        """
        :param flatten_embeddings: if True then model will return tensor with only present embedding,
        else also fake embeddings will be returned with patch statistics
        """

        super().__init__()
        self.flatten_embeddings = flatten_embeddings
        self.mlp = ProjectionHead(3, self.n_output_point, self.projection_dim)

    def get_patch_embeddings(self, x, labels):
        pooled, samples_idx, patch_idx = self.get_pooled_embeddings(x, labels)
        embs = pooled[samples_idx, :, patch_idx]
        embs = self.mlp(embs)
        return embs, samples_idx, patch_idx

    def get_pooled_embeddings(self, features, labels):
        # features = self.forward_features(x)
        one_hot = F.one_hot(labels).float()
        pooled = torch.bmm(features, one_hot)
        counts = one_hot.sum(dim=1)
        one = torch.ones(1, dtype=torch.float, device=features.device)
        counts_nonzero = torch.where(counts != 0, counts, one)
        pooled /= counts_nonzero.unsqueeze(1)
        batch_idx, patch_idx = torch.nonzero(counts, as_tuple=True)
        return pooled, batch_idx, patch_idx

    @abstractmethod
    def forward_features(self, x):
        pass

    def forward(self, x, patch_labels):
        return self.get_patch_embeddings(self.forward_features(x), patch_labels)
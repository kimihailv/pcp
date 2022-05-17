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
                self.layers.append(nn.Conv1d(in_channels, in_channels, 1))
                self.layers.append(nn.BatchNorm1d(in_channels))
                self.layers.append(nn.ReLU(True))
            else:
                self.layers.append(nn.Conv1d(in_channels, out_channels, 1, bias=False))
                self.layers.append(nn.BatchNorm1d(out_channels))

        self.layers = nn.ModuleList(self.layers)
        # self.pad_emb = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x, counts=None):
        '''if counts is not None:
            mask = counts == 0
            batch_idx, patch_idx = torch.nonzero(mask.int(), as_tuple=True)
            x[batch_idx, :, patch_idx] = self.pad_emb'''

        for layer in self.layers:
            x = layer(x)

        return x


class BaseModel(nn.Module, ABC):
    n_output_inst = -1
    n_output_point = -1
    projection_dim = 128

    def __init__(self, max_patch_count=-1, flatten_embeddings: bool = False):
        """
        :param flatten_embeddings: if True then model will return tensor with only present embedding,
        else also fake embeddings will be returned with patch statistics
        """

        super().__init__()
        self.flatten_embeddings = flatten_embeddings
        self.max_patch_count = max_patch_count
        self.mlp = ProjectionHead(3, self.n_output_point, self.projection_dim)

    def group_by(self, features, labels):
        max_patch_count = labels.max().item() + 1 if self.max_patch_count < 0 else self.max_patch_count
        mask = labels.unsqueeze(1) == torch.arange(max_patch_count).to(labels.device).unsqueeze(1)
        # batch_size x n_patches x feature_dim x n_features
        zero = torch.tensor([0], dtype=features.dtype).to(labels.device)
        grouped = torch.where(mask.unsqueeze(2), features.unsqueeze(1),
                              zero)
        return grouped, mask

    def get_patch_embeddings_onehot(self, x, labels):
        pooled, batch_idx, patch_idx = self.get_pooled_embeddings(x, labels)
        pooled = pooled[batch_idx, :, patch_idx]
        pooled = self.mlp(pooled.unsqueeze(2)).squeeze(2)
        return pooled, patch_idx

    def get_pooled_embeddings(self, x, labels):
        features = self.forward_features(x)
        one_hot = F.one_hot(labels).float()
        pooled = torch.bmm(features, one_hot)
        counts = one_hot.sum(dim=1)
        one = torch.ones(1, dtype=torch.float, device=features.device)
        counts_nonzero = torch.where(counts != 0, counts, one)
        pooled /= counts_nonzero.unsqueeze(1)
        batch_idx, patch_idx = torch.nonzero(counts, as_tuple=True)
        return pooled, batch_idx, patch_idx

    def get_patch_embeddings(self, features, labels):
        """
        :param features: features for all face/point/pixel/voxel, batch_size x feature_dim x n_features
        :param labels: patches/segmentation labels, batch_size x n_features
        """
        grouped, mask = self.group_by(features, labels)
        grouped = grouped.sum(dim=-1).transpose(2, 1).contiguous()
        counts = mask.sum(axis=-1)
        counts_nonzero = torch.where(counts != 0, counts, 1)
        pooled = grouped / counts_nonzero.unsqueeze(1)

        pooled = self.mlp(pooled, counts=counts)

        if self.flatten_embeddings:
            batch_idx, patch_idx = torch.nonzero(counts, as_tuple=True)
            return pooled[batch_idx, :, patch_idx]

        return pooled, counts

    @abstractmethod
    def forward_features(self, x):
        pass

    def forward(self, x, patch_labels):
        return self.get_patch_embeddings(self.forward_features(x), patch_labels)
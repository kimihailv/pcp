import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import h5py
from copy import deepcopy


# from faiss import PCAMatrix


def all_gather(x, dim):
    rank = dist.get_rank()
    sizes = [torch.zeros(1, device=rank) for _ in range(dist.get_world_size())]

    size = torch.ones(1, device=dist.get_rank()) * x.size(0)
    dist.all_gather(sizes, size)

    sizes = torch.cat(sizes, dim=0).long()
    max_size = sizes.max().item()
    idx = torch.cat((torch.zeros(1, device=rank, dtype=sizes.dtype), sizes.cumsum(dim=0)), dim=0)

    if dim == 3:
        tensors = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors, x)
        return torch.cat(tensors, dim=0), idx[rank], idx[rank + 1]

    elif dim == 2:
        padded = torch.empty(max_size, x.size(1),
                             dtype=x.dtype,
                             device=rank)
        padded[:x.size(0)] = x
        tensors = [torch.zeros(max_size, x.size(1), device=rank, dtype=x.dtype) for _ in range(dist.get_world_size())]
    else:
        padded = torch.empty(max_size,
                             dtype=x.dtype,
                             device=rank)
        padded[:x.size(0)] = x
        tensors = [torch.zeros(max_size, device=rank, dtype=x.dtype) for _ in range(dist.get_world_size())]

    dist.all_gather(tensors, padded)

    tensors = torch.cat(tensors, dim=0)
    slices = []

    for i, size in enumerate(sizes.tolist()):
        start_idx = i * max_size
        end_idx = start_idx + size
        slices.append(tensors[start_idx:end_idx])

    return torch.cat(slices), idx[rank], idx[rank + 1]


def shuffle(x):
    x, start_idx, end_idx = all_gather(x, dim=3)
    idx_shuffle = torch.randperm(x.shape[0]).to(dist.get_rank())
    torch.distributed.broadcast(idx_shuffle, src=0)
    idx_unshuffle = torch.argsort(idx_shuffle)
    idx_this = idx_shuffle[start_idx:end_idx]

    return x[idx_this], idx_unshuffle


def unshuffle(x, idx_unshuffle):
    x, start_idx, end_idx = all_gather(x, dim=3)
    idx_this = idx_unshuffle[start_idx:end_idx]
    return x[idx_this]


def prepare_queue_initialization(hdf5_path, n_classes, size_per_cls, emb_dim):
    with h5py.File(hdf5_path, 'r') as f:
        patch_embeddings = f['patch_embeddings'][:]
        k = np.where(f['k_range'][:] == n_classes)[0][0]
        patch_labels = f['patch_labels'][k]
        pca = PCAMatrix(patch_embeddings.shape[1], emb_dim)
        pca.train(patch_embeddings)
        patch_embeddings = pca.apply_py(patch_embeddings)
        labels = np.unique(patch_labels)
        data = []

        for l in labels:
            embs = patch_embeddings[patch_labels == l]
            embs_size = embs.shape[0]
            if embs_size < size_per_cls:
                pad_size = size_per_cls - embs.shape[0]
                pad = np.random.randn(pad_size, emb_dim)
                embs = np.concatenate((embs, pad), axis=0)
            elif embs_size > size_per_cls:
                perm = np.random.permutation(embs_size)[:size_per_cls]
                embs = embs[perm]

            data.append(embs)

        data = np.array(data, dtype=np.float32)

        return torch.from_numpy(data).transpose(2, 1).contiguous()


class SimSiam(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder.projection_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.encoder.projection_dim)
        )
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, q, k, labels):
        q_proj, flat_labels = self.encoder.get_patch_embeddings_onehot(q, labels)
        k_proj, _ = self.encoder.get_patch_embeddings_onehot(k, labels)

        q_pred = self.predictor(q_proj)
        k_pred = self.predictor(k_proj)

        loss = self.criterion(q_pred, k_proj.detach()) + self.criterion(k_pred, q_proj.detach())

        loss = -loss.mean() / 2
        return loss, {'cosine_sim': loss}


class MoCo(nn.Module):
    def __init__(self, encoder, q_size, temperature):
        super().__init__()
        self.size = q_size
        ptr = torch.zeros(1)
        self.register_buffer('ptr', ptr)
        q = F.normalize(torch.randn(encoder.projection_dim, self.size), dim=0)
        self.register_buffer('queue', q)
        self.encoder = encoder
        self.key = deepcopy(self.encoder)
        self.temperature = temperature
        self.beta = 0.999
        for p in self.key.parameters():
            p.requires_grad = False

    def forward(self, q, k, labels, update_q=False):
        q, flat_labels = self.encoder.get_patch_embeddings_onehot(q, labels)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            if update_q:
                self.update_momentum_encoder()
                # in train mode we use several gpus
                k, idx_unshuffle = shuffle(k)
                k, batch_idx, patch_idx = self.key.get_pooled_embeddings(k, labels)
                k = unshuffle(k, idx_unshuffle)
                k = self.key.mlp(k[batch_idx, :, patch_idx].unsqueeze(2)).squeeze(2)
            else:
                # validation mode
                k, _ = self.key.get_patch_embeddings_onehot(k, labels)

        k = F.normalize(k, dim=1)

        pos_logits = torch.einsum('nc,nc->n', q, k).unsqueeze(1)
        neg_logits = torch.einsum('nc,ck->nk', q, self.queue.detach().clone())

        logits = torch.cat((pos_logits, neg_logits), dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        if update_q:
            self._dequeue_and_enqueue(k)

        ce = F.cross_entropy(logits, labels)
        return ce, {'ce': ce}

    @torch.no_grad()
    def update_momentum_encoder(self):
        for ema_p, m_p in zip(self.key.parameters(), self.encoder.parameters()):
            ema_p.data.mul_(self.beta).add_(m_p.data, alpha=1 - self.beta)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.size:
            self.queue[:, ptr:] = keys.T[:, :self.size - ptr]
            written = self.size - ptr
            ptr = (ptr + batch_size) % self.size
            self.queue[:, :ptr] = keys.T[:, written:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.size  # move pointer

        self.ptr[0] = ptr


class SupQueue(nn.Module):
    def __init__(self, size_per_cls, n_classes, embedding_dim, negatives_per_cls, init_data_path):
        super().__init__()
        self.size_per_cls = size_per_cls
        self.n_classes = n_classes
        self.negatives_per_cls = negatives_per_cls
        self.embedding_dim = embedding_dim
        ptr = torch.zeros(n_classes, dtype=torch.long)
        self.register_buffer('ptr', ptr)

        if init_data_path is None:
            queue = torch.randn(n_classes, embedding_dim, self.size_per_cls)
            self.register_buffer('q', queue)
        else:
            queue = prepare_queue_initialization(init_data_path, n_classes, size_per_cls, self.embedding_dim)
            self.register_buffer('q', queue)

        self.q = F.normalize(self.q, dim=1)

    def forward(self, labels, return_only_positives=False):
        """
        :param labels: bs
        :return: tuple of tensors (positive_pairs, negative_pairs)
        """

        bs = labels.size(0)
        # bs x dim x size_per_cls
        positives = self.q[labels]

        if return_only_positives:
            return positives

        rng = torch.arange(self.n_classes, device=labels.device).unsqueeze(0).expand(bs, -1)
        neg_idx = ~F.one_hot(labels, self.n_classes).bool()
        neg_idx = rng[neg_idx].view(bs, -1)  # bs x (n_classes - 1)
        # bs x dim x (n_classes - 1) x size_per_cls
        negatives = self.q[neg_idx].permute(0, 2, 1, 3)
        perm = torch.randperm(self.size_per_cls, device=labels.device)[:self.negatives_per_cls].view(1, 1, 1, -1)
        negatives = torch.gather(negatives, 3,
                                 perm.expand(negatives.size(0), negatives.size(1), self.n_classes - 1, -1))

        return positives, negatives.view(bs, self.embedding_dim, -1)

    @torch.no_grad()
    def dequeue_enqueue(self, x, labels):
        x, _, _ = all_gather(x, dim=2)
        labels, _, _ = all_gather(labels, dim=1)
        all_labels = labels.unique(sorted=True).tolist()

        for label in all_labels:
            update = x[labels == label]
            ptr = self.ptr[label]
            upd_size = update.size(0)

            if ptr + upd_size > self.size_per_cls:
                self.q[label, :, ptr:] = update.T[:, :self.size_per_cls - ptr]
                written = self.size_per_cls - ptr
                ptr = (ptr + upd_size) % self.size_per_cls
                self.q[label, :, :ptr] = update.T[:, written:]
            else:
                self.q[label, :, ptr:ptr + upd_size] = update.T
                ptr = (ptr + upd_size) % self.size_per_cls

            self.ptr[label] = ptr


class SupCon(nn.Module):
    def __init__(self, encoder, q_size_per_cls, n_classes, negatives_per_cls, temperature, beta=0.999,
                 init_data_path=None):
        super().__init__()
        self.encoder = encoder
        self.key = deepcopy(self.encoder)
        self.beta = beta
        self.queue = SupQueue(q_size_per_cls, n_classes, self.encoder.projection_dim, negatives_per_cls,
                              init_data_path=init_data_path)
        self.temperature = temperature

        for p in self.key.parameters():
            p.requires_grad = False

    def forward(self, q, k, labels, update_q=False):
        q, flat_labels = self.encoder.get_patch_embeddings_onehot(q, labels)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            if update_q:
                self.update_momentum_encoder()
                # in train mode we use several gpus
                k, idx_unshuffle = shuffle(k)
                k, batch_idx, patch_idx = self.key.get_pooled_embeddings(k, labels)
                k = unshuffle(k, idx_unshuffle)
                k = self.key.mlp(k[batch_idx, :, patch_idx].unsqueeze(2)).squeeze(2)
            else:
                # validation mode
                k, _ = self.key.get_patch_embeddings_onehot(k, labels)

        k = F.normalize(k, dim=1)

        positives, negatives = self.queue(flat_labels)
        # bs x dim x (1 + size_per_cls)
        positives = torch.cat((k.unsqueeze(2), positives), dim=2)
        # positives = k.unsqueeze(2)

        pos_logits = torch.einsum("nc,nck -> nk", q, positives) / self.temperature
        neg_logits = torch.einsum("nc,nck -> nk", q, negatives) / self.temperature

        # bs x n_pos x (1 + n_neg)
        logits = torch.cat((pos_logits.unsqueeze(2),
                            neg_logits.unsqueeze(1).expand(-1, pos_logits.size(1), -1)),
                           dim=2)

        # target = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        # loss = F.cross_entropy(logits, target)
        denum = torch.logsumexp(logits, dim=2)
        loss = (denum - pos_logits).mean()

        if update_q:
            self.queue.dequeue_enqueue(k, flat_labels)
        return loss, {'ce': loss}

    @torch.no_grad()
    def update_momentum_encoder(self):
        for ema_p, m_p in zip(self.key.parameters(), self.encoder.parameters()):
            ema_p.data.mul_(self.beta).add_(m_p.data, alpha=1 - self.beta)


class NNSimSiam(nn.Module):
    def __init__(self, encoder, strategy, replace_rate, **queue_kwargs):
        super().__init__()
        self.encoder = encoder
        self.replace_rate = replace_rate
        if strategy.startswith('nn'):
            _, k_start, k_end = strategy.split('_')
            self.strategy = 'nn'
            self.k_start = int(k_start)
            self.k_end = int(k_end)
        elif strategy.startswith('int'):
            _, k = strategy.split('_')
            self.strategy = 'int'
            self.k = int(k)

        self.q = SupQueue(**queue_kwargs)
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder.projection_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.encoder.projection_dim)
        )

        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, x1, x2, labels):
        x1, batch_idx, patch_idx = self.encoder.get_pooled_embeddings(x1, labels)
        x1 = self.encoder.mlp(x1[batch_idx, :, patch_idx].unsqueeze(2)).squeeze(2)
        x2, _ = self.encoder.get_patch_embeddings_onehot(x2, labels)  # bs x dim

        x1_aug = self.make_augmentation(x1, patch_idx)
        x2_aug = self.make_augmentation(x2, patch_idx)
        x1_target = self.scatter(x1_aug, batch_idx)
        x2_target = self.scatter(x2_aug, batch_idx)

        x1_pred = self.predictor(self.scatter(x1, batch_idx))
        x2_pred = self.predictor(self.scatter(x2, batch_idx))

        loss = -self.criterion(x1_pred, x2_target) - self.criterion(x2_pred, x1_target)
        loss = loss.div(2).mean()

        self.q.dequeue_enqueue(x1, patch_idx)

        return loss, {'neg_cos_sim': loss}

    def scatter(self, x, labels):
        batch_idx_unique, counts = torch.unique(labels, return_counts=True)
        embeddings = torch.zeros(batch_idx_unique.size(0), x.size(1), dtype=torch.float32, device=x.device)
        embeddings.scatter_(0, labels.unsqueeze(1).expand(-1, x.size(1)), x)
        embeddings /= counts.unsqueeze(1)
        return embeddings

    def make_augmentation(self, q, labels):
        q = q.detach().clone()
        idx_to_replace = torch.randperm(q.size(0))[:int(self.replace_rate * q.size(0))]
        keys = self.q(labels[idx_to_replace], return_only_positives=True)

        dists = (q[idx_to_replace].unsqueeze(2) - keys).pow(2).sum(dim=1)

        if self.strategy == 'nn':
            if self.k_start != self.k_end:
                k = torch.randint(self.k_start, self.k_end, size=(1,)).item()
            else:
                k = self.k_start

            nn_idx = torch.topk(dists, dim=1, largest=False, k=k)[1][:, -1].unsqueeze(1)
            rep = torch.gather(keys, 2, nn_idx.unsqueeze(1).expand(-1, keys.size(1), -1)).squeeze(2)

        elif self.strategy == 'int':
            weights, nn_idx = torch.topk(dists, dim=1, largest=False, k=self.k)
            weights = 1 / (weights + 1e-8)
            weights /= weights.sum(dim=1, keepdim=True)
            # bs x dim x k
            nn = torch.gather(keys, 2, nn_idx.unsqueeze(1).expand(-1, keys.size(1), -1))
            rep = (nn * weights.unsqueeze(1)).sum(dim=2)

        q[idx_to_replace] = rep
        return q


class BYOL(nn.Module):
    def __init__(self, encoder, n_steps, task, tau_base=0.99):
        super().__init__()
        self.encoder = encoder
        self.target = deepcopy(encoder)
        self.tau_base = tau_base
        self.task = task
        self.n_steps = n_steps
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder.projection_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.encoder.projection_dim)
        )

        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward_one_view(self, x1, x2, labels):
        mask = None
        v1_features = self.encoder.forward_features(x1)
        v2_features = self.target.forward_features(x2)

        if self.task == 'local2global':
            v1_patches, samples_idx, patch_idx = self.encoder.get_patch_embeddings(v1_features, labels)
            v1_patches = self.predictor(v1_patches)
            v1_patches, mask = self.unflatten(v1_patches, samples_idx, patch_idx)
            v2_global = self.target.mlp(self.target.forward_instance(v2_features))
            loss = self.compute_loss(v1_patches, v2_global, mask)

        elif self.task == 'local2local':
            v1, _, _ = self.encoder.get_patch_embeddings(v1_features, labels)
            v1 = self.predictor(v1)
            v2, _, _ = self.target.get_patch_embeddings(v2_features, labels)
        else:
            v1 = self.encoder.mlp(self.encoder.forward_instance(v1_features))
            v1 = self.predictor(v1)
            v2 = self.target.mlp(self.target.forward_instance(v2_features))

        if self.task != 'local2global':
            loss = self.compute_loss(v1, v2, mask)

        return loss

    def forward(self, x1, x2, labels, step):
        self.update_target_net(step)
        loss = self.forward_one_view(x1, x2, labels) + self.forward_one_view(x2, x1, labels)
        loss = loss.mean()
        return loss, {'dist': loss}

    def compute_loss(self, v1, v2, mask):
        v1 = F.normalize(v1, dim=1)
        v2 = F.normalize(v2, dim=1)

        if self.task == 'local2global':
            # v1: bs x dim x n
            # v2: bs x dim
            v2 = v2.unsqueeze(2)

        cos_sim = (v1 * v2).sum(dim=1)
        loss = 2 - 2 * cos_sim

        if self.task == 'local2global':
            loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)

        return loss

    @staticmethod
    def unflatten(x, samples_idx, patch_labels):
        batch_size = samples_idx.max() + 1
        n_patches = patch_labels.max() + 1
        dim = x.size(1)
        idx = samples_idx * n_patches + patch_labels
        out = torch.zeros(batch_size * n_patches, dim, device=x.device, dtype=x.dtype)
        out = (out
               .scatter_(0, idx.unsqueeze(1).expand(-1, dim), x)
               .view(batch_size, n_patches, dim)
               .transpose(2, 1)
               .contiguous()
               )
        mask = torch.zeros(batch_size * n_patches, dtype=torch.float32, device=x.device)
        mask = mask.scatter_(0, idx, 1).view(batch_size, n_patches)
        return out, mask

    @torch.no_grad()
    def update_target_net(self, step):
        tau = 1 - (1 - self.tau_base) * (np.cos(np.pi * step / self.n_steps) + 1) / 2
        for tgt_p, onl_p in zip(self.target.parameters(), self.encoder.parameters()):
            tgt_p.data.mul_(tau).add_(onl_p.data, alpha=1 - tau)

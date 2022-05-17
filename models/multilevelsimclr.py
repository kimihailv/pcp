import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from enum import Enum
from collections import defaultdict


class LocalityLevel(Enum):
    Instance = 0
    Patch = 1
    Points = 2
    PointsPatch = 3
    InstancePatch = 4


class MultiLevelSimCLR(nn.Module):
    def __init__(self, encoder, locality_levels,
                 loss_locality_levels, temperature, sup_loss,
                 max_samples=100, sample_frac=1, multilayer=False, predict_pseudolabels=False):
        super().__init__()
        self.encoder = encoder
        self.locality_levels = self.from_strs_to_enum(locality_levels)
        self.temperature = temperature
        self.max_samples = max_samples
        self.multilayer = multilayer
        self.sample_frac = sample_frac
        self.loss = MultiLocalityContrastiveLoss(self.from_strs_to_enum(loss_locality_levels), temperature, sup_loss)
        if LocalityLevel.Instance in self.locality_levels:
            self.instance_proj = nn.Linear(self.encoder.n_output_inst, self.encoder.n_output_point, bias=False)

        if predict_pseudolabels:
            self.head = nn.Conv1d(self.encoder.n_output_inst, 60, 1)
        else:
            self.head = None

    def forward(self, x1, x2, patch_labels1):
        v1_embeddings = self.get_embeddings(x1, patch_labels1)
        v2_embeddings = self.get_embeddings(x2, patch_labels1)
        return self.loss.gather_and_forward(v1_embeddings, v2_embeddings)

    def get_embeddings(self, x, patch_labels):
        embeddings = {}
        features = self.encoder.forward_features(x)

        if LocalityLevel.Instance in self.locality_levels:
            instance_embeddings = self.encoder.forward_instance(features)
            instance_embeddings = self.instance_proj(instance_embeddings)
            embeddings[LocalityLevel.Instance] = [self.encoder.mlp(instance_embeddings)]

        if LocalityLevel.Patch in self.locality_levels:
            patch_embeddings, patch_samples_idx, flat_patch_labels = self.encoder.get_patch_embeddings(features,
                                                                                                       patch_labels)
            embeddings[LocalityLevel.Patch] = [patch_embeddings, patch_samples_idx, flat_patch_labels]

        if LocalityLevel.Points in self.locality_levels:
            points_features, points_samples_idx, points_labels = self.sample_features_from_patches(features,
                                                                                                   patch_labels)
            embeddings[LocalityLevel.Points] = [points_features, points_samples_idx, points_labels]

        if self.head is not None:
            embeddings['logits'] = (self.head(features), patch_labels)

        return embeddings

    def sample_features_from_patches(self, features, patch_labels):
        points_features = []
        samples_idx_new = []
        points_labels = []

        for sample_i, (sample_features, sample_labels) in enumerate(zip(features, patch_labels)):
            unique_labels = torch.unique(sample_labels)

            for label in unique_labels:
                w = (sample_labels == label).float()
                total_points = int(w.sum().item())
                to_sample = int(total_points * self.sample_frac)
                if to_sample == 0:
                    to_sample = total_points

                to_sample = min(to_sample, self.max_samples)
                idx = torch.multinomial(w, num_samples=to_sample)
                points_features.append(sample_features[:, idx])
                samples_idx_new.append(torch.full((to_sample,), sample_i, device=features.device))
                points_labels.append(torch.full((to_sample,), label, device=features.device))

        points_features = torch.cat(points_features, dim=1).T.contiguous()
        samples_idx_new = torch.cat(samples_idx_new, dim=0)
        points_labels = torch.cat(points_labels, dim=0)

        return self.encoder.mlp(points_features), samples_idx_new, points_labels

    def from_strs_to_enum(self, strs):
        enum_elems = []
        for str in strs:
            enum_elems.append(LocalityLevel[str])

        return enum_elems


class MultiLocalityContrastiveLoss(nn.Module):
    def __init__(self, locality_levels, temperature, sup_loss, weights=None):
        super().__init__()
        self.locality_levels = locality_levels
        self.temperature = temperature
        self.sup_loss = sup_loss

        if weights is None:
            self.weights = defaultdict(lambda: 1)
        else:
            self.weights = weights

    def forward(self, v1, v2, scale_factor=1):
        loss_dict = {}
        loss = 0

        if LocalityLevel.Instance in self.locality_levels:
            ins_loss = self.contrastive_loss(v1[LocalityLevel.Instance][0],
                                             v2[LocalityLevel.Instance][0],
                                             anchor_mode='all',
                                             support_mode='all').div_(scale_factor)
            loss_dict.update({'instance': ins_loss})
            loss += ins_loss * self.weights[LocalityLevel.Instance]

        if LocalityLevel.Patch in self.locality_levels:
            v1_patch_embeddings, samples_idx, patch_labels = v1[LocalityLevel.Patch]
            v2_patch_embeddings, _, _ = v2[LocalityLevel.Patch]

            negatives_mask = self.make_batch_constraint_mask(samples_idx, samples_idx).repeat(2, 2)

            if self.sup_loss:
                labels = patch_labels
            else:
                n_patches = patch_labels.max() + 1
                labels = samples_idx * n_patches + patch_labels

            patch_loss = self.contrastive_loss(v1_patch_embeddings,
                                               v2_patch_embeddings,
                                               anchor_mode='all',
                                               support_mode='all',
                                               labels1=labels,
                                               labels2=labels,
                                               negatives_mask=negatives_mask).div_(scale_factor)
            '''patch_loss = self.contrastive_loss_with_one_pos(v1_patch_embeddings,
                                                            v2_patch_embeddings,
                                                            anchor_mode='all',
                                                            support_mode='all',
                                                            sample_idx1=samples_idx,
                                                            sample_idx2=samples_idx,
                                                            labels1=labels,
                                                            labels2=labels,
                                                            discard_self_sim=True)'''

            loss_dict.update({'patch': patch_loss})
            loss += patch_loss * self.weights[LocalityLevel.Patch]

        if LocalityLevel.PointsPatch in self.locality_levels:
            pairs = zip(v1[LocalityLevel.Points], v2[LocalityLevel.Points])
            points_embeddings, points_samples_idx, points_labels = tuple(torch.cat(pair, dim=0) for pair in pairs)
            pairs = zip(v1[LocalityLevel.Patch], v2[LocalityLevel.Patch])
            patch_embeddings, patch_samples_idx, patch_labels = tuple(torch.cat(pair, dim=0) for pair in pairs)

            negatives_mask = self.make_batch_constraint_mask(points_samples_idx, patch_samples_idx)

            if self.sup_loss:
                labels1 = points_labels
                labels2 = patch_labels

            else:
                n_patches = patch_labels.max() + 1
                labels1 = points_samples_idx * n_patches + points_labels
                labels2 = patch_samples_idx * n_patches + patch_labels

            points_patch_loss = self.contrastive_loss(points_embeddings,
                                                      patch_embeddings,
                                                      anchor_mode='one',
                                                      support_mode='one',
                                                      labels1=labels1,
                                                      labels2=labels2,
                                                      logits_mask='full',
                                                      negatives_mask=negatives_mask).div_(scale_factor)

            '''points_patch_loss = self.contrastive_loss_with_one_pos(points_embeddings,
                                                                   patch_embeddings,
                                                                   anchor_mode='one',
                                                                   support_mode='one',
                                                                   sample_idx1=points_samples_idx,
                                                                   sample_idx2=patch_samples_idx,
                                                                   labels1=points_labels,
                                                                   labels2=patch_labels,
                                                                   discard_self_sim=False)'''

            loss_dict.update({'points_patch': points_patch_loss})
            loss += points_patch_loss * self.weights[LocalityLevel.PointsPatch]

        if LocalityLevel.Points in self.locality_levels:
            v1_points_embeddings, samples_idx, points_labels = v1[LocalityLevel.Points]
            v2_points_embeddings, _, _ = v2[LocalityLevel.Points]

            negatives_mask = self.make_batch_constraint_mask(samples_idx, samples_idx).repeat(2, 2)

            if self.sup_loss:
                labels = points_labels
            else:
                n_patches = points_labels.max() + 1
                labels = samples_idx * n_patches + points_labels

            points_loss = self.contrastive_loss(v1_points_embeddings,
                                                v2_points_embeddings,
                                                anchor_mode='all',
                                                support_mode='all',
                                                labels1=labels,
                                                labels2=labels,
                                                negatives_mask=negatives_mask).div_(scale_factor)

            loss_dict.update({'points': points_loss})
            loss += points_loss * self.weights[LocalityLevel.Points]

        if LocalityLevel.InstancePatch in self.locality_levels:
            ins_embeddings = torch.cat((v1[LocalityLevel.Instance][0], v2[LocalityLevel.Instance][0]), dim=0)
            ins_labels = torch.arange(ins_embeddings.size(0) // 2, device=ins_embeddings.device).repeat(2)
            pairs = zip(v1[LocalityLevel.Patch], v2[LocalityLevel.Patch])
            patch_embeddings, patches_samples_idx, _ = tuple(torch.cat(pair, dim=0) for pair in pairs)

            patch_ins_loss = self.contrastive_loss(patch_embeddings,
                                                   ins_embeddings,
                                                   anchor_mode='one',
                                                   support_mode='one',
                                                   labels1=patches_samples_idx,
                                                   labels2=ins_labels,
                                                   logits_mask='full').div_(scale_factor)

            loss_dict.update({'patch_ins': patch_ins_loss})
            loss += patch_ins_loss * self.weights[LocalityLevel.InstancePatch]

        if 'logits' in v1:
            logits1, labels1 = v1['logits']
            logits2, labels2 = v2['logits']

            ce = F.cross_entropy(logits1, labels1) + F.cross_entropy(logits2, labels2)
            ce.div_(2)
            loss_dict.update({'ce': ce})
            loss += ce

        return loss, loss_dict

    def gather_and_forward(self, v1, v2):
        v1_gathered = defaultdict(lambda: [])
        v2_gathered = defaultdict(lambda: [])

        for key in v1.keys():
            if key == LocalityLevel.Instance:
                v1_gathered[key].append(self.gather_equal_lengths(v1[key][0]))
                v2_gathered[key].append(self.gather_equal_lengths(v2[key][0]))
            elif key == 'logits':
                v1_gathered[key].append(self.gather_equal_lengths(v1[key][0]))
                v1_gathered[key].append(self.gather_equal_lengths(v1[key][1]))

                v2_gathered[key].append(self.gather_equal_lengths(v2[key][0]))
                v2_gathered[key].append(self.gather_equal_lengths(v2[key][1]))

            else:
                for i, v in enumerate((v1, v2)):
                    embeddings, samples_idx, labels = v[key]

                    embeddings = self.gather_var_lengths(embeddings)
                    samples_idx = self.gather_var_lengths(samples_idx, index_tensor=True)
                    labels = self.gather_var_lengths(labels)
                    if i == 0:
                        v1_gathered[key] = [embeddings, samples_idx, labels]
                    else:
                        v2_gathered[key] = [embeddings, samples_idx, labels]

        return self.forward(v1_gathered, v2_gathered)

    def contrastive_loss(self, v1_embeddings, v2_embeddings,
                         anchor_mode, support_mode,
                         logits_mask=None,
                         negatives_mask=None,
                         labels1=None, labels2=None):
        batch_size = v1_embeddings.size(0)
        if labels1 is None:
            mask = torch.eye(batch_size, device=v1_embeddings.device)
        else:
            mask = torch.eq(labels1.view(-1, 1), labels2.view(1, -1))

        v1_embeddings = F.normalize(v1_embeddings, dim=1)
        v2_embeddings = F.normalize(v2_embeddings, dim=1)

        if support_mode == 'one':
            support_repeat = 1
            embeddings = v2_embeddings
        elif support_mode == 'all':
            support_repeat = 2
            embeddings = torch.cat((v1_embeddings, v2_embeddings), dim=0)

        if anchor_mode == 'one':
            anchor_feature = v1_embeddings
            anchor_count = 1
        elif anchor_mode == 'all':
            anchor_feature = embeddings
            anchor_count = 2

        mask = mask.repeat(anchor_count, support_repeat).float()

        if logits_mask is None:
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(mask.device),
                0
            )  # discard self sim
        elif logits_mask == 'full':
            logits_mask = torch.ones_like(mask)

        mask *= logits_mask

        logits = anchor_feature @ embeddings.T / self.temperature

        negatives_mask = ~logits_mask.bool() if negatives_mask is None else ~logits_mask.bool() | negatives_mask
        logits.masked_fill_(negatives_mask & (~mask.bool()), -100)

        log_probs = logits - torch.logsumexp(logits, dim=1).unsqueeze(1)

        mean_over_pos = (log_probs * mask).sum(dim=1) / mask.sum(dim=1)
        loss = -mean_over_pos.mean()

        return loss

    def contrastive_loss_with_one_pos(self,
                                      v1_embeddings, v2_embeddings,
                                      anchor_mode, support_mode,
                                      sample_idx1, sample_idx2,
                                      labels1, labels2,
                                      discard_self_sim):
        batch_size = v1_embeddings.size(0)
        positive_mask = torch.eq(sample_idx1.view(-1, 1), sample_idx2.view(1, -1))  # true where samples' ids are equal
        denum_mask = torch.eq(labels1.view(-1, 1),
                              labels2.view(1, -1))  # mask to discard false negatives from denumerator

        v1_embeddings = F.normalize(v1_embeddings, dim=1)
        v2_embeddings = F.normalize(v2_embeddings, dim=1)

        if support_mode == 'one':
            support_repeat = 1
            embeddings = v2_embeddings
        elif support_mode == 'all':
            support_repeat = 2
            embeddings = torch.cat((v1_embeddings, v2_embeddings), dim=0)

        if anchor_mode == 'one':
            anchor_feature = v1_embeddings
            anchor_count = 1
        elif anchor_mode == 'all':
            anchor_feature = embeddings
            anchor_count = 2

        positive_mask = positive_mask.repeat(anchor_count, support_repeat).float()
        denum_mask = denum_mask.repeat(anchor_count, support_repeat)

        if discard_self_sim:
            logits_mask = torch.scatter(
                torch.ones_like(positive_mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(positive_mask.device),
                0
            )
        else:
            logits_mask = torch.ones_like(positive_mask)

        positive_mask *= logits_mask
        denum_mask |= ~logits_mask.bool()

        logits = anchor_feature @ embeddings.T / self.temperature
        logits.masked_fill_(denum_mask, -100)

        log_probs = logits - torch.logsumexp(logits, dim=1).unsqueeze(1)

        mean_over_pos = (log_probs * positive_mask).sum(dim=1) / positive_mask.sum(dim=1)
        loss = -mean_over_pos.mean()

        return loss

    @staticmethod
    def make_batch_constraint_mask(samples_idx1, samples_idx2):
        return torch.eq(samples_idx1.view(-1, 1), samples_idx2.view(1, -1))

    def contrastive_inside_sample(self, v1_embeddings, v2_embeddings, samples_idx, patch_labels):
        def unflatten(x):
            batch_size = samples_idx.max() + 1
            n_patches = patch_labels.max() + 1
            dim = x.size(1)
            idx = samples_idx * n_patches + patch_labels
            out = torch.zeros(batch_size * n_patches, dim, device=x.device, dtype=x.dtype)
            (out
             .scatter_(0, idx.unsqueeze(1).expand(-1, dim), x)
             .view(batch_size, n_patches, dim)
             .transpose(2, 1)
             .contiguous()
             )
            mask = torch.zeros(batch_size * n_patches, dtype=torch.bool, device=x.device)
            mask.scatter_(0, idx, True).view(batch_size, n_patches)
            return out, ~mask

        v1_embeddings = F.normalize(v1_embeddings, dim=1)
        v2_embeddings = F.normalize(v2_embeddings, dim=1)
        v1_embeddings, v1_mask = unflatten(v1_embeddings)
        v2_embeddings, v2_mask = unflatten(v2_embeddings)

        n_patches = v1_embeddings.size(2)

        embs = torch.cat((v1_embeddings, v2_embeddings), dim=2)
        # b x (2 * n_patches) x (2 * n_patches)
        logits = torch.bmm(embs.transpose(2, 1), embs) / self.temperature

        # discard self similarities
        mask = ~torch.eye(n_patches * 2, dtype=torch.bool, device=v1_embeddings.device)
        logits = (logits
                  .masked_select(mask)
                  .view(-1, 2 * n_patches, 2 * n_patches - 1)
                  .transpose(2, 1)
                  .contiguous()
                  )

        # ignore error for empty patches
        empty_patches_mask = torch.cat((v1_mask, v2_mask), dim=1)
        labels = torch.cat((torch.arange(n_patches) + n_patches - 1,
                            torch.arange(n_patches)
                            ), dim=0).to(v1_embeddings.device)
        ignore_label = torch.LongTensor([-100]).to(v1_embeddings.device)
        labels = torch.where(empty_patches_mask, ignore_label, labels)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def gather_equal_lengths(x):
        tensors = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors, x)
        tensors[dist.get_rank()] = x
        return torch.cat(tensors, dim=0)

    @staticmethod
    def gather_var_lengths(x, index_tensor=False):
        rank = dist.get_rank()
        sizes = [torch.zeros(1, device=rank) for _ in range(dist.get_world_size())]

        size = torch.ones(1, device=dist.get_rank()) * x.size(0)
        dist.all_gather(sizes, size)

        sizes = torch.cat(sizes, dim=0).long()
        max_size = sizes.max().item()
        shape = list(x.shape)
        shape[0] = max_size
        padded = torch.empty(*shape,
                             dtype=x.dtype,
                             device=rank)
        if index_tensor:
            offset = x.max() + 1
            offsets = [torch.zeros(1, device=rank, dtype=x.dtype) for _ in range(dist.get_world_size())]
            dist.all_gather(offsets, offset)
            offsets = torch.cat(offsets, dim=0)
            offsets = offsets.cumsum(dim=0)

            if rank > 0:
                x_shifted = x + offsets[rank - 1]
            else:
                x_shifted = x

            padded[:x.size(0)] = x_shifted

        else:
            padded[:x.size(0)] = x
        tensors = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors, padded)

        tensors = torch.cat(tensors, dim=0)
        slices = []

        for i, size in enumerate(sizes.tolist()):
            start_idx = i * max_size
            end_idx = start_idx + size
            slices.append(tensors[start_idx:end_idx])

        if not index_tensor:
            slices[rank] = x

        return torch.cat(slices, dim=0)

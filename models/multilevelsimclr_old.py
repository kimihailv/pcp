import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from enum import Enum
from collections import defaultdict
from .basemodel_old import ProjectionHead


class LocalityLevel(Enum):
    Instance = 0
    Patch = 1
    Points = 2
    PointsPatch = 3
    InstancePatch = 4


class MultiLevelSimCLR(nn.Module):
    def __init__(self, encoder, locality_levels, loss_locality_levels,
                 temperature, sup_loss, max_patch_count=-1,
                 max_samples=100, sample_frac=1, multilayer=False):
        super().__init__()
        self.encoder = encoder
        self.encoder.max_patch_count = max_patch_count
        self.locality_levels = self.from_strs_to_enum(locality_levels)
        self.temperature = temperature
        self.max_samples = max_samples
        self.multilayer = multilayer
        self.sample_frac = sample_frac
        self.loss = MultiLocalityContrastiveLoss(self.from_strs_to_enum(loss_locality_levels), temperature, sup_loss)
        self.instance_mlp = ProjectionHead(3, self.encoder.n_output_inst, self.encoder.projection_dim)

    def forward(self, x1, x2, patch_labels1):
        v1_embeddings = self.get_embeddings(x1, patch_labels1)
        v2_embeddings = self.get_embeddings(x2, patch_labels1)
        return self.loss.gather_and_forward(v1_embeddings, v2_embeddings)

    def get_embeddings(self, x, patch_labels):
        embeddings = {}
        features = self.encoder.forward_features(x)

        if LocalityLevel.Instance in self.locality_levels:
            instance_embeddings = self.encoder.forward_instance(features)
            embeddings[LocalityLevel.Instance] = [self.instance_mlp(instance_embeddings.unsqueeze(2)).squeeze(2)]

        if LocalityLevel.Patch in self.locality_levels:
            patch_embeddings, patch_sizes = self.encoder.get_patch_embeddings(features, patch_labels)
            embeddings[LocalityLevel.Patch] = [patch_embeddings, patch_sizes]

        if LocalityLevel.Points in self.locality_levels:
            projected_elementwise_features = self.encoder.mlp(features)
            grouped_proj_features, mask = self.encoder.group_by(projected_elementwise_features, patch_labels)
            point_embeddings, samples_sizes = self.sample_features_from_patches(grouped_proj_features, mask)
            embeddings[LocalityLevel.Points] = [point_embeddings, samples_sizes]

        return embeddings

    def sample_features_from_patches(self, grouped, mask):
        # bs x c x 2048
        # bs x 2048
        # batch_size x n_patches
        batched_patch_sizes = mask.sum(2)
        sampled_features = []

        actual_sample_sizes = torch.min((batched_patch_sizes * self.sample_frac).int(),
                                        torch.tensor([self.max_samples]).to(batched_patch_sizes.device))
        max_samples = self.max_samples  # actual_sample_sizes.max().item()
        fake_samples = torch.zeros(grouped.size(2), max_samples).to(grouped.device)

        for patches_features, sample_sizes, sample_weights in zip(grouped, actual_sample_sizes, mask.float()):
            # patches_features has shape n_patches x feature_dim x n_features
            # sample_sizes has shape n_patches
            # sample_weights has shape n_patches x n_features
            batch_sampled_features = []

            for patch_features, sample_size, sample_weight in zip(patches_features, sample_sizes, sample_weights):
                # patch_features has shape feature_dim x n_features
                # sample_size has shape 1
                # sample_weight has shape n_features
                if sample_size == 0:
                    batch_sampled_features.append(fake_samples)
                    continue

                selected_idx = torch.multinomial(sample_weight, sample_size.item())
                samples = patch_features.index_select(1, selected_idx)
                if sample_size < max_samples:
                    samples = F.pad(samples, (0, max_samples - sample_size))

                batch_sampled_features.append(samples)

            sampled_features.append(torch.stack(batch_sampled_features, dim=0))

        return torch.stack(sampled_features, dim=0), actual_sample_sizes

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
            ins_loss = self.instance_contrastive_loss(v1[LocalityLevel.Instance][0],
                                                      v2[LocalityLevel.Instance][0])
            ins_loss['instance_ce'].div_(scale_factor)
            loss_dict.update(ins_loss)
            loss += ins_loss['instance_ce'] * self.weights[LocalityLevel.Instance]

        if LocalityLevel.Patch in self.locality_levels:
            if not self.sup_loss:
                patch_loss = self.patch_contrastive_loss(v1[LocalityLevel.Patch],
                                                         v2[LocalityLevel.Patch])
            else:
                patch_loss = self.patch_contrastive_loss_sup(v1[LocalityLevel.Patch],
                                                             v2[LocalityLevel.Patch])
            patch_loss['patch_ce'].div_(scale_factor)
            loss_dict.update(patch_loss)
            loss += patch_loss['patch_ce'] * self.weights[LocalityLevel.Patch]

        if LocalityLevel.PointsPatch in self.locality_levels:
            v1_ = v1[LocalityLevel.Patch] + v1[LocalityLevel.Points]
            v2_ = v2[LocalityLevel.Patch] + v2[LocalityLevel.Points]
            points_patch_loss = self.point_patch_loss(v1_, v2_)

            for k in points_patch_loss:
                points_patch_loss[k].div_(scale_factor)

            loss_dict.update(points_patch_loss)
            sum_loss = 0

            for term in points_patch_loss.values():
                sum_loss += term

            loss += sum_loss * self.weights[LocalityLevel.PointsPatch]

        if LocalityLevel.InstancePatch in self.locality_levels:
            v1_ = v1[LocalityLevel.Instance] + v1[LocalityLevel.Patch]
            v2_ = v2[LocalityLevel.Instance] + v2[LocalityLevel.Patch]

            ins_patch_loss = self.instance_patch_loss(v1_, v2_)

            for k in ins_patch_loss:
                ins_patch_loss[k].div_(scale_factor)

            loss_dict.update(ins_patch_loss)
            sum_loss = 0

            for l in ins_patch_loss.values():
                sum_loss += l

            loss += sum_loss * self.weights[LocalityLevel.InstancePatch]

        return loss, loss_dict

    def gather_and_forward(self, v1, v2):
        v1_gathered = defaultdict(lambda: [])
        v2_gathered = defaultdict(lambda: [])

        for key in v1.keys():
            for tensor1, tensor2 in zip(v1[key], v2[key]):
                t1 = self.gather_concat(tensor1)
                t2 = self.gather_concat(tensor2)

                v1_gathered[key].append(t1)
                v2_gathered[key].append(t2)

        return self.forward(v1_gathered, v2_gathered)

    def instance_contrastive_loss(self, v1_embeddings, v2_embeddings):
        v1_embeddings = F.normalize(v1_embeddings, dim=1)
        v2_embeddings = F.normalize(v2_embeddings, dim=1)

        batch_size = v1_embeddings.size(0)
        embs = torch.cat((v1_embeddings, v2_embeddings), dim=0)
        logits = embs @ embs.transpose(1, 0) / self.temperature

        # discard self similarities
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=v1_embeddings.device)
        logits = (logits
                  .masked_select(mask)
                  .view(2 * batch_size, 2 * batch_size - 1)
                  .contiguous()
                  )

        labels = torch.cat((torch.arange(batch_size) + batch_size - 1,
                            torch.arange(batch_size)
                            ), dim=0).to(v1_embeddings.device)

        return {'instance_ce': F.cross_entropy(logits, labels)}

    def patch_contrastive_loss(self, x1, x2):
        """
        x1 and x2 are tuples with not flattened embeddings and patch_counts
        """
        v1_embeddings, v1_patch_counts = x1
        v2_embeddings, v2_patch_counts = x2

        n_patches = v1_embeddings.size(2)

        v1_embeddings = F.normalize(v1_embeddings, dim=1)
        v2_embeddings = F.normalize(v2_embeddings, dim=1)

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
        empty_patches_mask = torch.cat(((v1_patch_counts == 0), (v2_patch_counts == 0)), dim=1)
        labels = torch.cat((torch.arange(n_patches) + n_patches - 1,
                            torch.arange(n_patches)
                            ), dim=0).to(v1_embeddings.device)
        ignore_label = torch.LongTensor([-100]).to(v1_embeddings.device)
        labels = torch.where(empty_patches_mask, ignore_label, labels)

        return {'patch_ce': F.cross_entropy(logits, labels)}

    def patch_contrastive_loss_sup(self, x1, x2):
        v1_embeddings, v1_patch_counts = x1
        v2_embeddings, v2_patch_counts = x2

        batch_idx1, patch_idx1 = torch.nonzero(v1_patch_counts, as_tuple=True)
        batch_idx2, patch_idx2 = torch.nonzero(v2_patch_counts, as_tuple=True)

        v1_embeddings = v1_embeddings[batch_idx1, :, patch_idx1]
        v2_embeddings = v2_embeddings[batch_idx2, :, patch_idx2]
        embs = torch.cat((v1_embeddings, v2_embeddings), dim=0)

        mask = torch.eq(patch_idx1.view(-1, 1), patch_idx2.view(-1, 1).T).float()
        mask = mask.repeat(2, 2)
        self_sim_mask = 1 - torch.eye(embs.size(0), device=embs.device)
        mask *= self_sim_mask

        logits = embs @ embs.T / self.temperature
        logprobs = logits - torch.logsumexp(logits * self_sim_mask, dim=1).unsqueeze(1)

        mean_log_prob_pos = (mask * logprobs).sum(1) / mask.sum(1)
        return {'patch_ce': -mean_log_prob_pos.mean()}

    def point_patch_loss(self, x1, x2):
        """
        x1 and x2 are tuples of patch_embeddings, patch_sizes, point_embeddings, samples_sizes
        """

        def point_patch_loss_one_view(patch_embeddings, point_embeddings, sample_sizes):
            # patch_embeddings bs x dim x n_patches
            # point_embeddings bs x n_patches x dim x n_samples
            device = patch_embeddings.device
            batch_size, embedding_dim, n_patches = patch_embeddings.shape

            # bs x dim x (n_patches x n_samples)
            point_embeddings = point_embeddings.transpose(2, 1).contiguous().view(batch_size, embedding_dim, -1)

            # bs x n_patches (from patch_embeddings) x n_patches (from samples) x n_samples
            logits = (torch.bmm(patch_embeddings.transpose(2, 1), point_embeddings)
                      .contiguous()
                      .view(batch_size, n_patches, n_patches, -1)) / self.temperature

            # Point-patch contrast – points classification problem, where classes are patches' number
            # bs x n_patches x n_samples
            idx = (torch.arange(logits.size(-1), device=device)
                   .unsqueeze(0)
                   .unsqueeze(1)
                   .expand(batch_size, n_patches, -1)
                   )

            ignore_label = torch.LongTensor([-100]).to(device)
            labels = torch.where(idx < sample_sizes.unsqueeze(-1),
                                 torch.arange(n_patches, device=device).unsqueeze(-1),
                                 ignore_label)

            return F.cross_entropy(logits, labels)

        v1_patch_embeddings, v1_patch_sizes, v1_points_embeddings, v1_samples_sizes = x1
        v2_patch_embeddings, v2_patch_sizes, v2_points_embeddings, v2_samples_sizes = x2

        point_patch_v1_loss = point_patch_loss_one_view(v1_patch_embeddings, v1_points_embeddings, v1_samples_sizes)
        point_patch_v2_loss = point_patch_loss_one_view(v2_patch_embeddings, v2_points_embeddings, v2_samples_sizes)
        point_patch_v12_loss = point_patch_loss_one_view(v1_patch_embeddings, v2_points_embeddings, v2_samples_sizes)
        point_patch_v21_loss = point_patch_loss_one_view(v2_patch_embeddings, v1_points_embeddings, v1_samples_sizes)

        losses = {
            'point_patch_v1': point_patch_v1_loss,
            'point_patch_v2': point_patch_v2_loss,
            'point_patch_v12': point_patch_v12_loss,
            'point_patch_v21': point_patch_v21_loss
        }
        return losses

    def instance_patch_loss(self, x1, x2):
        # x1 and x2 are tuples of instance_embeddings, patch_embeddings, patch_sizes
        def instance_patch_loss_one_view(ins_embeddings, patch_embeddings, patch_sizes):
            bs, emb_dim, n_patches = patch_embeddings.shape

            ins_embeddings = F.normalize(ins_embeddings, dim=1)
            # bs * n_patches x emb_dim
            patch_embeddings = F.normalize(patch_embeddings, dim=1).transpose(2, 1).contiguous().view(-1, emb_dim)
            # bs * n_patches x bs
            logits = patch_embeddings @ ins_embeddings.t() / self.temperature
            labels = (torch.arange(bs, device=ins_embeddings.device)
                      .unsqueeze(1)
                      .expand(-1, n_patches)
                      .contiguous()
                      )

            labels.masked_fill_(patch_sizes == 0, -100)
            return F.cross_entropy(logits, labels.view(-1))

        v1_ins_embeddings, v1_patch_embeddings, v1_patch_sizes = x1
        v2_ins_embeddings, v2_patch_embeddings, v2_patch_sizes = x2

        ins_patch_v1_loss = instance_patch_loss_one_view(v1_ins_embeddings, v1_patch_embeddings, v1_patch_sizes)
        ins_patch_v2_loss = instance_patch_loss_one_view(v2_ins_embeddings, v2_patch_embeddings, v2_patch_sizes)
        ins_patch_v12_loss = instance_patch_loss_one_view(v1_ins_embeddings, v2_patch_embeddings, v2_patch_sizes)
        ins_patch_v21_loss = instance_patch_loss_one_view(v2_ins_embeddings, v1_patch_embeddings, v1_patch_sizes)

        losses = {
            'ins_patch_v1': ins_patch_v1_loss,
            'ins_patch_v2': ins_patch_v2_loss,
            'ins_patch_v12': ins_patch_v12_loss,
            'ins_patch_v21': ins_patch_v21_loss
        }
        return losses

    @staticmethod
    def gather_concat(x):
        tensors = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors, x)
        tensors[dist.get_rank()] = x
        return torch.cat(tensors, dim=0)

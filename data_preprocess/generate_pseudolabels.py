import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from ..datasets import ShapeNetDataset, PointCloudNormalize
from ..models.diffae import DiffAE
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pyclustering.cluster.elbow import elbow
from sklearn.cluster import KMeans
import numpy as np


def cluster_kmeans(f, random_seed):
    elbow_instance = elbow(f, 2, 6)
    elbow_instance.process()
    k = elbow_instance.get_amount()
    return KMeans(n_clusters=k, random_state=random_seed).fit_predict(f)


def get_patch_embeddings(features, labels, batch_n, batch_size):
    batch_offset = batch_n * batch_size
    # bs x n_pts x n_patches
    patch_enc = F.one_hot(labels.long(), num_classes=6).to(features.device)
    # bs x dim x n_patches
    patch_embs = torch.bmm(features, patch_enc.float())

    sizes = patch_enc.sum(dim=1, keepdim=True)
    mask = sizes == 0
    sizes_non_zero = sizes.masked_fill(mask, 1)
    patch_embs /= sizes_non_zero
    # patch_embs = F.normalize(patch_embs, dim=1)

    all_embs = []
    mapping = []

    for i, (embs, m) in enumerate(zip(patch_embs.cpu(), ~mask[:, 0, :])):
        all_embs += [e for e in embs[:, m].t().cpu().numpy()]
        patch_idxs = torch.nonzero(m, as_tuple=True)[0].tolist()
        sample_idx = i + batch_offset
        mapping += list(zip([sample_idx for _ in range(len(patch_idxs))], patch_idxs))

    return all_embs, mapping


def convert_labels(patch_labels, point_labels, mapping):
    labels = np.zeros_like(point_labels)

    for (sample_idx, patch_idx), patch_label in zip(mapping, patch_labels):
        labels[sample_idx][point_labels[sample_idx] == patch_idx] = patch_label

    return labels


@torch.no_grad()
def get_features(model, x):
    t = torch.ones(x.size(0), dtype=torch.int64, device=x.device)
    time_emb = model.get_time_embeddings(t)
    xt = model.deterministic_forward_process(x, False, 2)
    out = model.ae(x, xt, time_emb, return_features=True)
    features = torch.cat(out[0][1], dim=1).cpu()
    return features


def get_labels_for_batch(features):
    def worker(feature, random_seed=4242):
        return cluster_kmeans(feature, random_seed)

    features = features.transpose(2, 1).contiguous().numpy()

    tasks = (
             delayed(worker)(features[sample_idx])
             for sample_idx in range(features.shape[0])
            )

    labels = np.array(Parallel(n_jobs=-1, backend='loky')(tasks))
    return torch.from_numpy(labels)


def generate_labels_and_patches(dataset_path, model):
    dataset = ShapeNetDataset(dataset_path, ['train', 'val', 'test'], ['all'],
                              transform=PointCloudNormalize(mode='shape_unit'))
    loader = DataLoader(dataset, batch_size=40, shuffle=False)
    points_local_labels = []
    patch_embeddings = []
    patch_points_map = []

    for batch_idx, (x, _) in tqdm(enumerate(loader), total=len(loader)):
        features = get_features(model, x.to(device))
        x[:, 0, :] *= -1
        flipped_features = get_features(model, x.to(device))
        features += flipped_features
        features /= 2
        labels = get_labels_for_batch(features)
        embeddings, mapping = get_patch_embeddings(features, labels, batch_idx, loader.batch_size)
        points_local_labels.append(labels.numpy())
        patch_embeddings += embeddings
        patch_points_map += mapping

    points_local_labels = np.concatenate(points_local_labels, axis=0)
    patch_embeddings = np.array(patch_embeddings)

    return points_local_labels, patch_embeddings, patch_points_map


def parse_args():
    parser = ArgumentParser(description='Labels generation by diffusion model')

    parser.add_argument('--weights',
                        action='store',
                        type=str,
                        help='path to weights')

    parser.add_argument('--dataset',
                        action='store',
                        type=str,
                        help='path to dataset')

    parser.add_argument('--output_file',
                        action='store',
                        type=str,
                        help='path to output file')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()
    device = 'cuda:0'
    model = DiffAE(total_steps=150).to(device)
    state = torch.load(opt.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    points_local_labels, patch_embeddings, patch_points_map = generate_labels_and_patches(opt.dataset, model)
    np.savez(f'{opt.output_file}',
             points_local_labels=points_local_labels,
             patch_embeddings=patch_embeddings,
             patch_points_map=patch_points_map)

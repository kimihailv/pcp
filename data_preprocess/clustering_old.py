import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from ..datasets import ShapeNetDataset
from ..models.diffusion import NoisePredictor
from ..models.diffusion import DiffusionModel
from sklearn.metrics import silhouette_score
from pyclustering.cluster.elbow import elbow
from argparse import ArgumentParser
from collections import Counter
from torch.utils.data import DataLoader
from faiss import Kmeans
import numpy as np


def cluster_kmeans(f, random_seed):
    k = elbow(f, kmin=4, kmax=10).process().get_amount()
    clt = Kmeans(f.shape[1], k, niter=100, nredo=3, spherical=False, seed=random_seed)
    clt.train(f)
    _, sample_labels = clt.index.search(f, 1)
    return sample_labels[:, 0], k


def worker(sample_idx, features, random_seed=4242):
    best_score = -1
    best_labels = None
    best_level = -1
    best_k = -1
    for feature_level in range(2, len(features)):
        f = features[feature_level][sample_idx]
        sample_labels, k = cluster_kmeans(f, random_seed)
        score = silhouette_score(f, sample_labels)

        if score > best_score:
            best_score = score
            best_labels = sample_labels
            best_level = feature_level
            best_k = k

    return best_labels, best_k, best_level


def get_labels_for_batch(x, model):
    features = model.get_features(x.to(device), [1])[1]
    features = [f.cpu().transpose(2, 1).contiguous().numpy() for f in features]
    # features.append(np.concatenate(features, axis=2))

    tasks = (delayed(worker)(sample_idx, features) for sample_idx in range(x.shape[0]))
    result = Parallel(n_jobs=-1, backend='loky')(tasks)

    labels = []
    k_stats = []
    level_stats = []

    for best_labels, best_k, best_level in result:
        labels.append(best_labels)
        k_stats.append(best_k)
        level_stats.append(best_level)

    return labels, k_stats, level_stats


def generate_labels(dataset_path, model, random_seed=4242):
    torch.manual_seed(random_seed)
    dataset = ShapeNetDataset(dataset_path, ['val'], ['all'])
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_labels = []
    k_stats = Counter()
    levels_stats = Counter()
    scores = []
    for x, _ in tqdm(loader):
        labels, best_ks, best_levels = get_labels_for_batch(x, model)
        all_labels += labels
        k_stats.update(best_ks)
        levels_stats.update(best_levels)

    stats = {
        'k': k_stats,
        'levels': levels_stats,
        'scores': scores
    }
    return np.array(all_labels), stats


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

    parser.add_argument('--feature_level',
                        action='store',
                        type=int,
                        help='features from which layer will be used, integer from 0 to 2')

    parser.add_argument('--min_k',
                        action='store',
                        type=int,
                        help='minimum k for KMeans')

    parser.add_argument('--max_k',
                        action='store',
                        type=int,
                        help='maximum k for KMeans')

    parser.add_argument('--k_step',
                        action='store',
                        type=int,
                        help='step size in range(min_k, max_k)')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()
    device = 'cuda:0'
    model = DiffusionModel(NoisePredictor(1024, 256),
                           4000,
                           time_embedding_dim=256).to(device)
    state = torch.load(opt.weights)
    model.load_state_dict(state)
    model.eval()

    print('Generating labels with KMeans')
    labels, stats = generate_labels(opt.dataset, model)
    np.savez(f'{opt.output_file}', labels=labels, stats=stats)

import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from ..datasets import ShapeNetDataset
from ..models.diffusion import NoisePredictor
from ..models.diffusion import DiffusionModel
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from faiss import Kmeans
import numpy as np


def cluster_kmeans(f, k, random_seed):
    clt = Kmeans(f.shape[1], k, niter=100, nredo=3, spherical=False, seed=random_seed)
    clt.train(f)
    _, sample_labels = clt.index.search(f, 1)
    return sample_labels[:, 0]


def worker(sample_idx, features, k_range, random_seed=4242):
    labels = [cluster_kmeans(features[sample_idx], k, random_seed) for k in k_range]
    return np.array(labels)


def get_labels_for_batch(x, model, feature_level, min_k, max_k, k_step):
    features = model.get_features(x.to(device), [1])[1]
    features = [f.cpu().transpose(2, 1).contiguous().numpy() for f in features]
    features.append(np.concatenate(features, axis=2))

    tasks = (
             delayed(worker)(sample_idx, features[feature_level], range(min_k, max_k, k_step))
             for sample_idx in range(x.shape[0])
            )

    return Parallel(n_jobs=-1, backend='loky')(tasks)


def generate_labels(dataset_path, model, feature_level, min_k, max_k, k_step, random_seed=4242):
    torch.manual_seed(random_seed)
    dataset = ShapeNetDataset(dataset_path, ['val'], ['all'])
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_labels = []
    for x, _ in tqdm(loader):
        labels = get_labels_for_batch(x, model, feature_level, min_k, max_k, k_step)
        all_labels += labels

    return np.stack(all_labels)


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
                        help='features from which layer will be used, integer from 0 to 3')

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
    labels = generate_labels(opt.dataset, model, opt.feature_level, opt.min_k, opt.max_k, opt.k_step)
    np.savez(f'{opt.output_file}', labels=labels)

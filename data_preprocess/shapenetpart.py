import numpy as np
import h5py
import torch
import multiprocessing as mp
from argparse import ArgumentParser
from json import load
from functools import partial
from tqdm.auto import tqdm


def sample(x, n_points):
    device = x.device
    B, C, N = x.shape
    centroids = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = x[batch_indices, :, farthest].view(B, C, 1)
        dist = torch.sum((x - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return torch.gather(x, 2, centroids.unsqueeze(1).expand(-1, C, -1)).cpu().numpy(), centroids.cpu().numpy()


def read_and_sample(ids, root_dir, n_points):
    def f(idx):
        category, idx = idx.strip().split('/')[1:]
        file_path = f'{root_dir}/{category}/{idx}.txt'
        data = np.loadtxt(file_path)
        pc = torch.from_numpy(data[:, :3].T[np.newaxis, ...].astype(np.float32)).to('cuda:0')

        if pc.size(2) < n_points:
            print(idx, pc.size(2))
            centroids = np.random.choice(pc.size(2), size=(n_points,), replace=True)
            pc = pc.cpu().numpy()
            pc = pc[:, :, centroids]
            centroids = centroids[np.newaxis, ...]
        else:
            pc, centroids = sample(pc, n_points)

        return pc, int(category), data[:, -1][centroids[0]], idx

    pcs, cats, labels, shape_ids = [], [], [], []

    for idx in ids:
        pc, cat, l, shape_id = f(idx)
        pcs.append(pc)
        cats.append(cat)
        labels.append(l)
        shape_ids.append(shape_id)

    return np.concatenate(pcs, axis=0), np.array(cats), np.vstack(labels), shape_ids


def process(root_dir, ids_file, n_points, batch_size):
    worker = partial(read_and_sample,
                     root_dir=root_dir,
                     n_points=n_points)

    with open(ids_file, 'r') as ids:
        files_list = load(ids)
        # total_len = len(files_list) // batch_size + (len(files_list) % batch_size != 0)
        batches = (files_list[i:i+batch_size] for i in range(0, len(files_list), batch_size))
        pcs, cats, labels, shape_ids = [], [], [], []

        with mp.Pool(10) as pool:
            for pc, cat, l, shape_id in pool.imap_unordered(worker, batches):
                pcs.append(pc)
                cats.append(cat)
                labels.append(l)
                shape_ids += shape_id
        return np.concatenate(pcs, axis=0), np.concatenate(cats, axis=0), np.concatenate(labels, axis=0), np.array(shape_ids, dtype='S')


def write_to_hdf5(root_dir, n_points, batch_size, output_file):
    with h5py.File(output_file, 'w') as f:
        for split in ('train', 'val', 'test'):
            ids_file = f'{root_dir}/train_test_split/shuffled_{split}_file_list.json'
            points, cats, labels, shape_ids = process(root_dir, ids_file, n_points, batch_size)
            split_group = f.create_group(split)
            split_group.create_dataset('points', data=points, dtype='f4', compression='gzip')
            split_group.create_dataset('categories', data=cats, dtype='i4', compression='gzip')
            split_group.create_dataset('labels', data=labels, dtype='i4', compression='gzip')
            split_group.create_dataset('shapes_ids', shape_ids.shape, data=shape_ids)


def parse_args():
    parser = ArgumentParser(description='ShapeNetPart preparation')

    parser.add_argument('--root_dir',
                        action='store',
                        type=str,
                        help='path to directory with data')

    parser.add_argument('--n_points',
                        action='store',
                        type=int,
                        help='the number of point to sample')

    parser.add_argument('--output_file',
                        action='store',
                        type=str,
                        help='path to output h5 file')

    parser.add_argument('--batch_size',
                        action='store',
                        type=int,
                        help='size of batch for fps')

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    write_to_hdf5(opts.root_dir, opts.n_points, opts.batch_size, opts.output_file)

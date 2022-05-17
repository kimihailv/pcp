import numpy as np
import torch
import h5py
import argparse
from tqdm.auto import tqdm

cat2label = {
    'airplane': 0,
    'bathtub': 1,
    'bed': 2,
    'bench': 3,
    'bookshelf': 4,
    'bottle': 5,
    'bowl': 6,
    'car': 7,
    'chair': 8,
    'cone': 9,
    'cup': 10,
    'curtain': 11,
    'desk': 12,
    'door': 13,
    'dresser': 14,
    'flower_pot': 15,
    'glass_box': 16,
    'guitar': 17,
    'keyboard': 18,
    'lamp': 19,
    'laptop': 20,
    'mantel': 21,
    'monitor': 22,
    'night_stand': 23,
    'person': 24,
    'piano': 25,
    'plant': 26,
    'radio': 27,
    'range_hood': 28,
    'sink': 29,
    'sofa': 30,
    'stairs': 31,
    'stool': 32,
    'table': 33,
    'tent': 34,
    'toilet': 35,
    'tv_stand': 36,
    'vase': 37,
    'wardrobe': 38,
    'xbox': 39
}


def read_one_file(root_dir, idx):
    idx = idx.strip()
    parts = idx.split('_')
    shape_id = int(parts[-1])
    category = '_'.join(parts[:-1])
    shape_path = f'{root_dir}/{category}/{idx}.txt'
    pts = np.loadtxt(shape_path, delimiter=',').astype(np.float32)[:, :3].T
    return pts, shape_id, cat2label[category]


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

    return torch.gather(x, 2, centroids.unsqueeze(1).expand(-1, C, -1)).cpu().numpy()


def process(root_dir, ids_file, n_points, batch_size):
    with open(ids_file, 'r') as ids:
        ids = list(ids.readlines())
        samples = []
        shapes_ids = []
        labels = []

        total_batches = len(ids) // batch_size + int(len(ids) % batch_size != 0)
        for start in tqdm(range(0, len(ids), batch_size), total=total_batches):
            points = []
            for idx in ids[start:start+batch_size]:
                pts, shape_id, label = read_one_file(root_dir, idx)
                points.append(pts)
                shapes_ids.append(shape_id)
                labels.append(label)

            points = torch.from_numpy(np.stack(points)).to('cuda:0')
            samples.append(sample(points, n_points))

        return np.concatenate(samples, axis=0), np.array(shapes_ids), np.array(labels)


def write_to_hdf5(root_dir, n_points, batch_size, output_file):
    with h5py.File(output_file, 'w') as f:
        for split in ('train', 'test'):
            ids_file = f'{root_dir}/modelnet40_{split}.txt'
            samples, shapes_ids, labels = process(root_dir, ids_file, n_points, batch_size)
            split_group = f.create_group(split)
            split_group.create_dataset('points', data=samples, dtype='f4', compression='gzip')
            split_group.create_dataset('shapes_ids', data=shapes_ids, dtype='i4', compression='gzip')
            split_group.create_dataset('labels', data=labels, dtype='i4', compression='gzip')


def parse_args():
    parser = argparse.ArgumentParser(description='ModelNet40 preparation')

    parser.add_argument('--root_dir',
                        action='store',
                        type=str,
                        help='path to directory with data')

    parser.add_argument('--n_points',
                        action='store',
                        type=int,
                        help='the number of point to sample (max 10k)')

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

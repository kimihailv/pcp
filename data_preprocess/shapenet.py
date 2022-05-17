import h5py
import numpy as np
import argparse
import point_cloud_utils as pcu
from glob import glob
from collections import defaultdict
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed
from pickle import load

synsets = [2691156, 2747177, 2773838, 2801938, 2808440, 2818832,
           2828884, 2843684, 2871439, 2876657, 2880940, 2924116,
           2933112, 2942699, 2946921, 2954340, 2958343, 2992529,
           3001627, 3046257, 3085013, 3207941, 3211117, 3261776,
           3325088, 3337140, 3467517, 3513137, 3593526, 3624134,
           3636649, 3642806, 3691459, 3710193, 3759954, 3761084,
           3790512, 3797390, 3928116, 3938244, 3948459, 3991062,
           4004475, 4074963, 4090263, 4099429, 4225987, 4256520,
           4330267, 4379243, 4401088, 4460130, 4468005, 4530566,
           4554684]


def process(file_path, n_points):
    parts = file_path.split('/')
    synset = int(parts[-4])
    shape_id = parts[-3]
    v, f = pcu.load_mesh_vf(file_path)
    f_i, bc = pcu.sample_mesh_poisson_disk(v, f, int(n_points * 1.2))
    points = pcu.interpolate_barycentric_coords(f, f_i, bc, v)

    if points.shape[0] > n_points:
        perm = np.random.choice(points.shape[0], n_points, replace=False)
        return points[perm].T, synset, shape_id

    return points.T, synset, shape_id


def write_to_npz(dst_dir, points, synset, shape_id):
    file_name = f'{synset}_{shape_id}.npz'
    np.savez(f'{dst_dir}/{file_name}', points=points)


def worker(dst_dir, n_points, file_path):
    try:
        points, synset, shape_id = process(file_path, n_points)
        write_to_npz(dst_dir, points, synset, shape_id)
    except:
        print(file_path)


def write_to_h5(src_dir, split_mapping, output_file):
    points_collections = {'train': defaultdict(lambda: []),
                          'val': defaultdict(lambda: []),
                          'test': defaultdict(lambda: [])}

    shape_ids_collections = {'train': defaultdict(lambda: []),
                             'val': defaultdict(lambda: []),
                             'test': defaultdict(lambda: [])}

    with h5py.File(output_file, 'w') as f:
        for split in ('train', 'val', 'test'):
            split_group = f.create_group(split)
            for synset in synsets:
                split_group.create_group(str(synset))

        file_paths = glob(f'{src_dir}/*.npz')
        for path in file_paths:
            parts = path.split('/')
            synset, shape_id = parts[-1][:-4].split('_')
            if shape_id not in split_mapping:
                continue
            split = split_mapping[shape_id]
            points_collections[split][synset].append(np.load(path)['points'])
            shape_ids_collections[split][synset].append(shape_id)

        for split, synset2points in points_collections.items():
            for synset, points in synset2points.items():
                f[split][synset].create_dataset('data', data=np.stack(points),
                                                dtype='f4', compression='gzip')
                f[split][synset].create_dataset('shape_ids',
                                                data=np.array(shape_ids_collections[split][synset], dtype='S'))


def parse_args():
    parser = argparse.ArgumentParser(description='ShapeNet55 preparation')

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

    parser.add_argument('--splits_path',
                        action='store',
                        type=str,
                        help='path to splits mapping')

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()

    working_dir = '/'.join(opts.output_file.split('/')[:-1])
    tmp_dir = Path(f'{working_dir}/tmp')
    # tmp_dir.mkdir(exist_ok=True, parents=True)

    worker_fn = partial(worker,
                        dst_dir=str(tmp_dir),
                        n_points=opts.n_points)

    file_paths = glob(f'{opts.root_dir}/*/*/models/*.obj')

    # Parallel(n_jobs=-1, backend='multiprocessing', verbose=50)(delayed(worker_fn)(file_path=path) for path in file_paths)
    split_mapping = load(open(opts.splits_path, 'rb'))
    write_to_h5(str(tmp_dir), split_mapping, opts.output_file)

import h5py
import numpy as np
from json import load
from torch.utils.data import Dataset
import random

synset2label = {
    2691156: 0,
    2773838: 1,
    2954340: 2,
    2958343: 3,
    3001627: 4,
    3261776: 5,
    3467517: 6,
    3624134: 7,
    3636649: 8,
    3642806: 9,
    3790512: 10,
    3797390: 11,
    3948459: 12,
    4099429: 13,
    4225987: 14,
    4379243: 15
}


class ShapeNetPartDataset2(Dataset):
    def __init__(self, h5_path, split, transform=None):
        super().__init__()
        self.data = h5py.File(h5_path, 'r')[split]
        self.points = self.data['points'][:]
        self.categories = self.data['categories'][:]
        self.labels = self.data['labels'][:]
        self.transform = transform

    def __getitem__(self, idx):
        points = self.points[idx]

        if self.transform is not None:
            points = self.transform(points.T).T.astype('float32')

        categories = self.categories[idx]
        labels = self.labels[idx]

        return points, synset2label[categories], labels

    def __len__(self):
        return self.points.shape[0]


class ShapeNetPartDataset(Dataset):
    def __init__(self, root_dir, split, n_points, transform=None, use_cache=True, sample_frac=1, seed=13214214):
        split_file_path = f'{root_dir}/train_test_split/shuffled_{split}_file_list.json'

        with open(split_file_path, 'r') as f:
            self.ids = load(f)

        self.ids = [idx.strip().split('/')[1:] for idx in self.ids]

        if sample_frac < 1:
            random.seed(seed)
            random.shuffle(self.ids)
            n_samples = int(len(self.ids) * sample_frac)
            self.ids = self.ids[:n_samples]

        self.root_dir = root_dir
        self.n_points = n_points
        self.transform = transform
        self.cache = {}
        self.use_cache = use_cache

    def __getitem__(self, i):
        if self.use_cache and i in self.cache:
            pc, category, labels = self.cache[i]
            return pc.T, category, labels

        category, idx = self.ids[i]
        data_file = f'{self.root_dir}/{category}/{idx}.txt'
        data = np.loadtxt(data_file, dtype=np.float32)

        if 0 < self.n_points != data.shape[0]:
            choice = np.random.choice(data.shape[0], self.n_points, replace=True)
            pc = data[choice, :3]
            labels = data[choice, -1]
        else:
            pc = data[:, :3]
            labels = data[:, -1]

        if self.transform is not None:
            pc = self.transform(pc).astype(np.float32)

        category = synset2label[int(category)]
        if self.use_cache:
            self.cache[i] = (pc, category, labels)
        return pc.T, category, labels

    def __len__(self):
        return len(self.ids)

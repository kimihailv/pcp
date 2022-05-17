import h5py
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class ABCDataset(Dataset):
    def __init__(self, hdf5_path, split, target, transform=None, sample_frac=1, seed=42):
        super().__init__()
        data = h5py.File(hdf5_path)
        self.points = data['points'][:]
        self.target = data[target][:]
        self.transform = transform
        train_ids, test_ids = train_test_split(np.arange(data['points'].shape[0]), train_size=0.7, random_state=42,
                                               shuffle=True)

        if split == 'train':
            ids = train_ids
        else:
            ids = test_ids

        self.points = self.points[ids]
        self.target = self.target[ids]

        if sample_frac < 1:
            n_samples = int(self.points.shape[0] * sample_frac)
            r = np.random.RandomState(seed)
            idx = r.permutation(self.points.shape[0])[:n_samples]
            self.points = self.points[idx]
            self.target = self.target[idx]

    def __getitem__(self, idx):
        pts = self.points[idx]

        if self.transform is not None:
            pts = self.transform(pts)

        return pts.T.astype('float32'), self.target[idx]

    def __len__(self):
        return self.points.shape[0]

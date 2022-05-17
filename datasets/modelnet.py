import h5py
import numpy as np
from torch.utils.data import Dataset

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
label2cat = {v: k for k, v in cat2label.items()}


class ModelNetDataset(Dataset):
    def __init__(self, h5_path, split, transform=None, few_shot='', classes=None):
        super().__init__()
        data = h5py.File(h5_path, 'r')[split]
        self.points = data['points'][:]
        self.labels = data['labels'][:]

        if len(few_shot) != 0:
            r = np.random.RandomState(42)
            way, shot = few_shot.split('_')
            way = int(way)
            shot = int(shot)
            self.classes = r.choice(40, way, replace=False) if classes is None else classes
            self.classes.sort()
            self.points, self.labels = self.pick_samples_from_classes(self.classes, shot, r)

        self.transform = transform

    def __getitem__(self, idx):
        pts = self.points[idx]
        if self.transform is not None:
            pts = self.transform(pts.T).T.astype(np.float32)

        return pts, self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]

    def pick_samples_from_classes(self, classes, n_samples, r):
        points = []
        labels = []

        for i, cls in enumerate(classes):
            mask = self.labels == cls
            assert mask.sum() > n_samples
            idx = r.choice(np.nonzero(mask)[0], n_samples, replace=False)
            points.append(self.points[idx])
            labels += [i for _ in range(n_samples)]

        return np.concatenate(points, axis=0), np.array(labels)

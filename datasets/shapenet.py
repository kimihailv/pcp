import h5py
import numpy as np
from torch.utils.data import Dataset

category2synset = {
    'airplane': '2691156',
    'ashcan': '2747177',
    'bag': '2773838',
    'basket': '2801938',
    'bathtub': '2808440',
    'bed': '2818832',
    'bench': '2828884',
    'birdhouse': '2843684',
    'bookshelf': '2871439',
    'bottle': '2876657',
    'bowl': '2880940',
    'bus': '2924116',
    'cabinet': '2933112',
    'camera': '2942699',
    'can': '2946921',
    'cap': '2954340',
    'car': '2958343',
    'chair': '3001627',
    'clock': '3046257',
    'computer keyboard': '3085013',
    'dishwasher': '3207941',
    'display': '3211117',
    'earphone': '3261776',
    'faucet': '3325088',
    'file': '3337140',
    'guitar': '3467517',
    'helmet': '3513137',
    'jar': '3593526',
    'knife': '3624134',
    'lamp': '3636649',
    'laptop': '3642806',
    'loudspeaker': '3691459',
    'mailbox': '3710193',
    'microphone': '3759954',
    'microwave': '3761084',
    'motorcycle': '3790512',
    'mug': '3797390',
    'piano': '3928116',
    'pillow': '3938244',
    'pistol': '3948459',
    'pot': '3991062',
    'printer': '4004475',
    'remote control': '4074963',
    'rifle': '4090263',
    'rocket': '4099429',
    'skateboard': '4225987',
    'sofa': '4256520',
    'stove': '4330267',
    'table': '4379243',
    'telephone': '4401088',
    'cellular telephone': '2992529',
    'tower': '4460130',
    'train': '4468005',
    'vessel': '4530566',
    'washer': '4554684'
}

synset2label = {
    '2691156': 0,
    '2747177': 1,
    '2773838': 2,
    '2801938': 3,
    '2808440': 4,
    '2818832': 5,
    '2828884': 6,
    '2843684': 7,
    '2871439': 8,
    '2876657': 9,
    '2880940': 10,
    '2924116': 11,
    '2933112': 12,
    '2942699': 13,
    '2946921': 14,
    '2954340': 15,
    '2958343': 16,
    '3001627': 17,
    '3046257': 18,
    '3085013': 19,
    '3207941': 20,
    '3211117': 21,
    '3261776': 22,
    '3325088': 23,
    '3337140': 24,
    '3467517': 25,
    '3513137': 26,
    '3593526': 27,
    '3624134': 28,
    '3636649': 29,
    '3642806': 30,
    '3691459': 31,
    '3710193': 32,
    '3759954': 33,
    '3761084': 34,
    '3790512': 35,
    '3797390': 36,
    '3928116': 37,
    '3938244': 38,
    '3948459': 39,
    '3991062': 40,
    '4004475': 41,
    '4074963': 42,
    '4090263': 43,
    '4099429': 44,
    '4225987': 45,
    '4256520': 46,
    '4330267': 47,
    '4379243': 48,
    '4401088': 49,
    '2992529': 50,
    '4460130': 51,
    '4468005': 52,
    '4530566': 53,
    '4554684': 54
}


class ShapeNetDataset(Dataset):
    def __init__(self, hdf5_path, splits, categories, transform=None, points_labels_path=None,
                 point_labels_level='global', n_classes=50):
        super().__init__()
        self.point_clouds = []
        self.labels = []
        self.points_labels = []

        if 'all' in categories:
            categories = category2synset.keys()

        prev_size = 0
        self.split_sizes = {}

        if points_labels_path is not None:
            points_labels_data = h5py.File(points_labels_path, 'r')
            point_labels_level = f'{point_labels_level}_labels'
            k = np.where(points_labels_data['k_range'][:] == n_classes)[0][0]

        with h5py.File(hdf5_path, 'r') as f:
            for split in splits:
                for category in categories:
                    synset = category2synset[category]
                    shapes_count = f[split][synset]['data'].shape[0]
                    self.labels += [synset2label[synset] for _ in range(shapes_count)]
                    data = f[split][synset]['data'][:]
                    self.point_clouds.append(data)

                if points_labels_path is not None:
                    if point_labels_level.startswith('global'):
                        self.points_labels.append(points_labels_data[split][point_labels_level][k].astype(np.int64))
                    else:
                        self.points_labels.append(points_labels_data[split][point_labels_level][:].astype(np.int64))

                self.split_sizes[split] = len(self.labels) - prev_size
                prev_size = len(self.labels)

        self.transform = transform
        self.point_clouds = np.concatenate(self.point_clouds, axis=0)
        if len(self.points_labels) != 0:
            self.points_labels = np.concatenate(self.points_labels, axis=0)
        self.points_labels_path = points_labels_path

    def __getitem__(self, idx):
        pc = self.point_clouds[idx]

        if self.transform is not None:
            pc = self.transform(pc.T).T.astype(np.float32)

        if self.points_labels_path is None:
            return pc, self.labels[idx]

        return pc, self.points_labels[idx]

    def __len__(self):
        return self.point_clouds.shape[0]

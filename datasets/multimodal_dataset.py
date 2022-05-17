from typing import *
from torch.utils.data import Dataset
from enum import Enum
from .datasets_meta import DatasetMeta
from abc import abstractmethod
import h5py
import numpy as np


class Modality(Enum):
    MESH = 'mesh'
    POINT_CLOUD = 'point_cloud'
    DEPTH_IMG = 'depth_images'
    SDF = 'sdf'


modality_mapping = {
    'mesh': Modality.MESH,
    'point_cloud': Modality.POINT_CLOUD,
    'depth_images': Modality.DEPTH_IMG,
    'sdf': Modality.SDF
}


class OneModalityDataset:
    def __init__(self, modality, transform, return_seg_labels):
        self.modality = modality
        self.transform = transform
        self.return_seg_labels = return_seg_labels

    @abstractmethod
    def get_item(self, item, labels_offset):
        pass


class PointCloudDataset(OneModalityDataset):
    def __init__(self, **base_dataset_kwargs):
        super().__init__(Modality.POINT_CLOUD, **base_dataset_kwargs)

    def get_item(self, item, labels_offset=0):
        point_cloud = item[self.modality.value]['data'][:]
        if self.transform is not None:
            point_cloud = self.transform(point_cloud)

        seg_labels = item[self.modality.value]['seg_labels'][:] + labels_offset if self.return_seg_labels else None
        return point_cloud.T.astype(np.float32), item[self.modality.value]['patch_labels'][:], seg_labels


class MeshNetDataset(OneModalityDataset):
    def __init__(self, max_faces, **base_dataset_kwargs):
        super().__init__(Modality.MESH, **base_dataset_kwargs)
        self.max_faces = max_faces

    def get_item(self, item, labels_offset=0):
        features = item[self.modality.value]['data']['features'][:]
        neighbors = item[self.modality.value]['data']['neighbors'][:].astype('int64')

        if self.transform is not None:
            features = self.transform(features)

        num_faces = features.shape[0]
        padding_width = self.max_faces - num_faces

        if num_faces < self.max_faces:
            if padding_width > num_faces:
                replace_sample = True
            else:
                replace_sample = False

            fill_idx = np.random.choice(num_faces, padding_width, replace=replace_sample)
            features = np.concatenate((features, features[fill_idx]))
            neighbors = np.concatenate((neighbors, neighbors[fill_idx]))

        features = np.transpose(features, (1, 0)).astype('float32')
        centers, corners, normals = features[:3], features[3:12], features[12:]
        corners = corners - np.concatenate([centers, centers, centers], 0)

        patch_labels = np.pad(item[self.modality.value]['patch_labels'][:], (0, padding_width),
                              mode='constant', constant_values=-1)

        seg_labels = None
        if self.return_seg_labels:
            seg_labels = np.pad(item[self.modality.value]['seg_labels'][:] + labels_offset,
                                (0, padding_width), mode='constant', constant_values=-1)

        return (centers, corners, normals, neighbors), patch_labels, seg_labels


class MultiModalDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str,
                 datasets: List[OneModalityDataset],
                 dataset_meta: DatasetMeta,
                 return_inst_label: bool = False,
                 ):
        """
        :param data_path: path to hdf5 file with data
        :param split: split of dataset, can be train/val/test or trainval
        :param datasets: set of datasets which will be used
        :param dataset_meta: meta information of dataset
         default – all categories will be selected
        :param return_inst_label: if true than instance label will be returned
        """
        self.dataset_meta = dataset_meta
        self.datasets = datasets
        self.return_inst_label = return_inst_label

        self.data = h5py.File(data_path, 'r')

        if not isinstance(split, list):
            splits = [split]
        else:
            splits = split
        keys = []
        for s in splits:
            keys_w_prefix = [f'{s}|{key}' for key in self.data[s].keys()]
            keys += keys_w_prefix

        # key of item has structure: {split}|{item_idx}_{instance_label?}

        self.keys = keys
        # shuffle(self.keys)
        
    def __getitem__(self, idxs):
        if isinstance(idxs, int):
            return self._getitem(idxs)
        elif isinstance(idxs, slice):
            start, stop, step = idxs.start, idxs.stop, idxs.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self.keys)
            if step is None:
                step = 1

            indexer = range(start, stop, step)
        elif isinstance(idxs, Iterable):
            indexer = idxs
        else:
            raise ValueError
        
        items = []
        for idx in indexer:
            items.append(self._getitem(idx))
        return items

    def _getitem(self, idx):
        features = []
        patch_labels_collection = []
        seg_labels_collection = []

        split, sample_idx, label = self._parse_key(self.keys[idx])
        for dataset in self.datasets:
            item = self.data[split][sample_idx]

            if self.dataset_meta.n_classes > 1 and hasattr(self.dataset_meta, 'offsets'):
                offset = self.dataset_meta.offsets[label]
            else:
                offset = 0

            data, patch_labels, seg_labels = dataset.get_item(item, offset)

            features.append(data)
            patch_labels_collection.append(patch_labels)

            if self.dataset_meta.seg_labels:
                seg_labels_collection.append(seg_labels)

        if self.return_inst_label and self.dataset_meta.n_classes > 1:
            label = self.dataset_meta.label2num[label]
        else:
            label = 0

        if len(self.datasets) == 1:
            features = features[0]
            patch_labels_collection = patch_labels_collection[0]
            if self.dataset_meta.seg_labels:
                seg_labels_collection = seg_labels_collection[0]

        if self.return_inst_label:
            if self.dataset_meta.seg_labels:
                return features, patch_labels_collection, seg_labels_collection, label
            return features, patch_labels_collection, label

        if self.dataset_meta.seg_labels:
            return features, patch_labels_collection, seg_labels_collection

        return features, patch_labels_collection

    def _parse_key(self, key):
        # key of item has structure: {split}|{item_idx}_{instance_label?}
        parts = key.split('|')
        if self.n_classes > 1:
            split, idx_label = parts
            idx, label = idx_label.split('_')
            return split, idx_label, label

        return parts[0], parts[1], None

    @property
    def n_classes(self):
        return self.dataset_meta.n_classes

    @property
    def cat_names(self):
        return self.dataset_meta.cat2label.keys()

    @property
    def n_parts(self):
        return self.dataset_meta.n_parts

    def __len__(self):
        return len(self.keys)


class DoubleDataset:
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    def __getitem__(self, idx):
        return self.base_dataset[idx], self.base_dataset[idx]

    def __len__(self):
        return len(self.base_dataset)

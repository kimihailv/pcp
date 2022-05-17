class DatasetMeta:
    dataset_name = None
    n_classes = None
    n_parts = None
    cat2label = None
    label2num = None
    n_patches = None
    seg_labels = False


class CosegMeta(DatasetMeta):
    dataset_name = 'coseg_aliens'
    n_classes = 1
    n_parts = 4
    cat2label = {
        'Aliens': '0'
    }
    label2num = {
        '0': 0
    }
    n_patches = 5
    seg_labels = True


class ABCMeta(DatasetMeta):
    dataset_name = 'abc'
    n_classes = 0
    n_patches = 5


datasets_meta_dict = {
    'coseg': CosegMeta,
    'abc': ABCMeta
}
import os.path as osp

from utils.data.dataset_catalog import COMMON_DATASETS

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Available datasets
_DATASETS = {
    'dota-v1-quad-five_train': {
        _IM_DIR:
            _DATA_DIR + '/DOTA/dota_800_200/train/images',
        _ANN_FN:
            _DATA_DIR + '/DOTA/dota_800_200/train/dota_800_200_train_five.json',
    },
    'dota-v1-quad-five_val': {
        _IM_DIR:
            _DATA_DIR + '/DOTA/dota_800_200/val/images',
        _ANN_FN:
            _DATA_DIR + '/DOTA/dota_800_200/val/dota_800_200_val_five.json',
    },
    'dota-v1-quad_vis': {
        _IM_DIR:
            _DATA_DIR + '/DOTA/dota_800_200/val/vis_images',
        _ANN_FN:
            _DATA_DIR + '/DOTA/dota_800_200/val/dota_800_200_val_vis.json',
    },
    'dota-v1-1024_trainval': {
        _IM_DIR:
            _DATA_DIR + '/DOTA/dota_1024_200/trainval_1024/images',
        _ANN_FN:
            _DATA_DIR + '/DOTA/dota_1024_200/trainval_1024/DOTA_trainval_1024.json',
    },
    'dota-v1-1024_test': {
        _IM_DIR:
            _DATA_DIR + '/DOTA/dota_1024_200/test_1024/images',
        _ANN_FN:
            _DATA_DIR + '/DOTA/dota_1024_200/test_1024/DOTA_test_1024.json',
    },
    'dota-v1-1024_trainval_ms': {
        _IM_DIR:
            _DATA_DIR + '/DOTA/dota_1024_200/trainval_1024_ms/images',
        _ANN_FN:
            _DATA_DIR + '/DOTA/dota_1024_200/trainval_1024_ms/DOTA_trainval_1024_ms.json',
    },
    'dota-v1-1024_test_ms': {
        _IM_DIR:
            _DATA_DIR + '/DOTA/dota_1024_200/test_1024_ms/images',
        _ANN_FN:
            _DATA_DIR + '/DOTA/dota_1024_200/test_1024_ms/DOTA_test_1024_ms.json',
    },
    'dota-v1-vis': {
        _IM_DIR:
            _DATA_DIR + '/DOTA/dota_800_200/patchs800',
        _ANN_FN:
            _DATA_DIR + '/DOTA/dota_800_200/train800.json',
    },
}
_DATASETS.update(COMMON_DATASETS)


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return _DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name][_ANN_FN]

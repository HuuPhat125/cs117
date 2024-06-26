# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.hico_det
import numpy as np

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))

# Set up hico_det for default setting
for split in ['train2015','test2015']:
    name = 'hico_det_{}'.format(split)
    __sets[name] = (lambda split=split, obj_id=None, obj_name=None:
                    datasets.hico_det(split, obj_id, obj_name))

# Set up hico_det for KO setting
# file_obj = './fast-rcnn/data/hico/list_object_class'
# list_obj = [line.strip() for line in open(file_obj)]

# for idx, obj_name in enumerate(list_obj):
#     obj_id = '{:02d}'.format(idx+1)
#     for split in ['train2015','test2015']:
#         name = 'hico_det_{}_{}_{}'.format(split, obj_id, obj_name)
#         __sets[name] = (lambda split=split, obj_id=obj_id, obj_name=obj_name:
#                         datasets.hico_det(split, obj_id, obj_name))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

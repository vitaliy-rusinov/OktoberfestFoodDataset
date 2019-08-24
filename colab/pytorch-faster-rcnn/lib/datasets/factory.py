# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco

import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['first9_test','first9_test_large','first9_test_nightmode', 'first9_trainval', '5_test',
'5_test_large','5_test_nightmode','5_trainval','10_test','10_test_large','10_test_nightmode',
'10_trainval','20_test','20_test_large','20_test_nightmode','20_trainval','40_test',
'40_test_large','40_test_nightmode','40_trainval','60_test','60_test_large','60_test_nightmode',
'60_trainval','80_test','80_test_large','80_test_nightmode','80_trainval','daynight_test',
'daynight_test_large','daynight_test_nightmode','daynight_trainval','test','test_large','test_nightmode',
'trainval','10_vid_test','10_vid_test_nightmode','10_vid_test_small','10_vid_trainval',
'40_vid_test','40_vid_test_nightmode','40_vid_test_small','40_vid_trainval',
'60_vid_test','60_vid_test_nightmode','60_vid_test_small','60_vid_trainval',
'80_vid_test','80_vid_test_nightmode','80_vid_test_small','80_vid_trainval',
]:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}_diff'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

_C.testing_mode = 'test_one_epoch_one_lang'
_C.resume_dir = ''
_C.resume_epoch = 'train-9'
_C.results_file = 'res'
_C.model_name = "Twitter/twhin-bert-base"
_C.batch_size = 16
_C.data_func = ''


def get_cfg_defaults():
  return _C.clone()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()


_C.resume_dir = ''
_C.resume_epoch = ''
_C.epochs = 1
_C.rounds = 2
_C.clients_num = 5
_C.model_name = "Twitter/twhin-bert-base"
_C.learning_rate = 5e-5
_C.batch_size = 16
_C.saving_folder_name = "trial_one"
_C.data_function = 'load_iid_data'


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
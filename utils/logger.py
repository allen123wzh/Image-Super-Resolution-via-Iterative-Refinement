import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime
import sys
import yaml


def setup_logger(name, local_rank, phase, 
                 level=logging.INFO, save_path=None, print=False):
    '''maskrcnn style: https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/utils/logger.py#L7'''
    l = logging.getLogger(name)
    l.setLevel(level)
    
    # No logging for non-master process
    if local_rank>0:
        return l
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    # FileHandle, save logs to disk
    if save_path:
        fh = logging.FileHandler(os.path.join(save_path, f'{phase}.log'))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        l.addHandler(fh)
    
    # StreamHandler, print to console
    if print:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        l.addHandler(ch)
    
    return l


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg
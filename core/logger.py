import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime
import sys

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse(args):
    phase = args.phase
    opt_path = args.config
    # gpu_ids = args.gpu_ids
    enable_wandb = args.enable_wandb
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    experiments_root = os.path.join(
        'experiments', '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    # change dataset length limit
    opt['phase'] = phase

    # export CUDA_VISIBLE_DEVICES
    # if gpu_ids is not None:
    #     opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
    #     gpu_list = gpu_ids
    # else:
    #     gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    # # debug
    # if 'debug' in opt['name']:
    #     opt['train']['val_freq'] = 2
    #     opt['train']['print_freq'] = 2
    #     opt['train']['save_checkpoint_freq'] = 3
    #     opt['datasets']['train']['batch_size'] = 2
    #     opt['model']['beta_schedule']['train']['n_timestep'] = 10
    #     opt['model']['beta_schedule']['val']['n_timestep'] = 10
    #     opt['datasets']['train']['data_len'] = 6
    #     opt['datasets']['val']['data_len'] = 3

    # # validation in train phase
    # if phase == 'train':
    #     opt['datasets']['val']['data_len'] = 6

    # W&B Logging
    try:
        log_wandb_ckpt = args.log_wandb_ckpt
        opt['log_wandb_ckpt'] = log_wandb_ckpt
    except:
        pass
    try:
        log_eval = args.log_eval
        opt['log_eval'] = log_eval
    except:
        pass
    try:
        log_infer = args.log_infer
        opt['log_infer'] = log_infer
    except:
        pass
    opt['enable_wandb'] = enable_wandb
    
    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


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

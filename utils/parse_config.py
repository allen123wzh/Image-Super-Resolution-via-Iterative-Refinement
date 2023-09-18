import os
import yaml
from datetime import datetime


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse_config(args):
    phase = args.phase
    opt_path = args.config

    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)

    # set log directory
    experiments_root = os.path.join('experiments', f'{opt["name"]}_{get_timestamp()}')
    opt['path']['experiments_root'] = experiments_root
    
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    # change dataset length limit
    opt['phase'] = phase

    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    # # W&B Logging
    # try:
    #     log_wandb_ckpt = args.log_wandb_ckpt
    #     opt['log_wandb_ckpt'] = log_wandb_ckpt
    # except:
    #     pass
    # try:
    #     log_eval = args.log_eval
    #     opt['log_eval'] = log_eval
    # except:
    #     pass
    # try:
    #     log_infer = args.log_infer
    #     opt['log_infer'] = log_infer
    # except:
    #     pass
    # opt['enable_wandb'] = enable_wandb
    
    opt = dict_to_nonedict(opt)

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



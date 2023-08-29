import torch
import numpy as np
import random
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/inf_ll_ddpm.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    opt['local_rank']=0

    # logging
    set_seed(42)
    
    # Root logger 
    logger = Logger.setup_logger(None, local_rank=opt['local_rank'], phase='train', 
                                 level=logging.INFO, save_path=opt['path']['log'], print=True)
    # Validation logger, saves val result to another file, no print
    logger_val =  Logger.setup_logger('val', local_rank=opt['local_rank'], phase='val', 
                                 level=logging.INFO, save_path=opt['path']['log'], print=False)

    # Logger.setup_logger(None, opt['path']['log'],
    #                     'train', level=logging.INFO, screen=True)
    # Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    # logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    avg_psnr = 0.0
    avg_ssim = 0.0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
        eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

        avg_psnr += eval_psnr
        avg_ssim += eval_ssim

        if wandb_logger and opt['log_eval']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)

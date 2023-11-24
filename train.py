import torch
import torch.distributed as dist

import argparse
import logging
import dataset as Data
import model as Model
from utils import *
from tensorboardX import SummaryWriter
import os
import numpy as np
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='config/m3fd_ir_rgb_6M.yaml',
    #                     help='JSON file for configuration')
    # parser.add_argument('-c', '--config', type=str, default='config/ll_bgd1x_ffhq_256.yaml',
    #                     help='JSON file for configuration')
    parser.add_argument('-c', '--config', type=str, default='config/debug_256.yaml',
                            help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                            help='Run either train(training) or val(generation)', default='train')

    # parse configs
    args = parser.parse_args()
    opt = parse_config(args)

    opt['local_rank']=-1
    
    # DDP initialization
    if len(opt['gpu_ids'])>1:
        opt['distributed']=True
        opt['datasets']['train']['distributed']=True
        
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        
        opt['local_rank']=device_id
        # opt['datasets']['train']['local_rank']=device_id
        print(f'Initialized DDP on rank {rank}')

        set_seed(42 + rank)
    else:
        set_seed(42)

    if opt['ir']:
        opt['datasets']['train']['ir'] = True
        opt['datasets']['val']['ir'] = True

    # Root logger 
    logger = setup_logger(None, local_rank=opt['local_rank'], phase='train', 
                                    level=logging.INFO, save_path=opt['path']['log'], print=True)
    # Validation logger, saves val result to another file, no print
    logger_val =  setup_logger('val', local_rank=opt['local_rank'], phase='val', 
                                    level=logging.INFO, save_path=opt['path']['log'], print=False)
    logger.info(dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            if opt['local_rank'] in {-1, 0}:
                val_set = Data.create_dataset(dataset_opt, phase)
                val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    if opt['phase'] == 'train':
        logger.info('Start training')
        while current_step < n_iter:
            current_epoch += 1
            if opt['distributed']:
                train_loader.sampler.set_epoch(current_epoch)            

            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters(current_step, 
                                              grad_accum=opt['train']['grad_accum'])
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

            if opt['local_rank'] in {-1,0}:
            # validation
                if current_epoch % opt['train']['val_epoch_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_psnr_ema = 0.0
                    avg_ssim_ema = 0.0
                    idx = 0
                    result_path = f'{opt["path"]["results"]}/{current_epoch}'

                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = tensor2img(visuals['SR'])  # uint8, super-res img
                        hr_img = tensor2img(visuals['HR'])  # uint8, GT hi-res
                        lr_img = tensor2img(visuals['LR'])  # uint8, Orig low-rs
                        if 'IR' in visuals:
                            ir_img = tensor2img(visuals['IR']) # uint8, IR
                        if 'SR_EMA' in visuals:
                            sr_ema_img = tensor2img(visuals['SR_EMA'])  # uint8, super-res img


                        save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        if 'IR' in visuals:
                            save_img(ir_img, '{}/{}_{}_ir.png'.format(result_path, current_step, idx))  # uint8, IR image
                        if 'SR_EMA' in visuals:
                            save_img(sr_ema_img, '{}/{}_{}_sr_ema.png'.format(result_path, current_step, idx))  # uint8, IR image

                        if 'IR' in visuals:
                            tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                # np.transpose(np.concatenate(
                                #     (lr_img, ir_img, hr_img, sr_img), axis=1), [2, 0, 1]),
                                # idx)
                                np.transpose(np.concatenate(
                                    (lr_img, hr_img, sr_img), axis=1), [2, 0, 1]),
                                idx)
                        else:
                            tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                np.transpose(np.concatenate(
                                    (lr_img, hr_img, sr_img), axis=1), [2, 0, 1]),
                                idx)
                        avg_psnr += calculate_psnr(sr_img, hr_img)
                        avg_ssim += calculate_ssim(sr_img, hr_img)
                        if 'SR_EMA' in visuals:
                            avg_psnr_ema += calculate_psnr(sr_ema_img, hr_img)
                            avg_ssim_ema += calculate_ssim(sr_ema_img, hr_img)                        

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    avg_psnr_ema = avg_psnr_ema / idx
                    avg_ssim_ema = avg_ssim_ema / idx
                    
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    # logger.info('# Validation # PSNR: {:.4e} # SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                    # logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                    # logger_val = logging.getLogger('val')  # validation logger
                    logger.info('# Validation #')
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e} EMA'.format(
                        current_epoch, current_step, avg_psnr_ema, avg_ssim_ema))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    tb_logger.add_scalar('psnr_ema', avg_psnr_ema, current_step)
                    tb_logger.add_scalar('ssim_ema', avg_ssim_ema, current_step)


                if current_epoch % opt['train']['save_ckpt_epoch_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = tensor2img(visuals['HR'])  # uint8
            lr_img = tensor2img(visuals['LR'])  # uint8
            fake_img = tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    save_img(tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = tensor2img(visuals['SR'])  # uint8
                save_img(sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                save_img(tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            # Metrics.save_img(
            #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = calculate_psnr(tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = calculate_ssim(tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        # logger.info('# Validation # PSNR: {:.4e} # SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
        # logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation #')
        # logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

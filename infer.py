import argparse
import logging
import dataset as Data
import model as Model
from utils import *
from tensorboardX import SummaryWriter
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/infer_ir_25M.yaml',
                        help='YAML file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val', 'test'], default='test')

    # parse configs
    args = parser.parse_args()
    opt = parse_config(args)

    opt['local_rank']=-1
    
    set_seed(42)

    if opt['ir']:
        opt['datasets']['test']['ir'] = True

    # Root logger 
    logger = setup_logger(None, local_rank=opt['local_rank'], phase=opt['phase'], 
                                    level=logging.INFO, save_path=opt['path']['log'], print=True)
    logger.info(dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        dataset = Data.create_dataset(dataset_opt, phase)
        dataloader = Data.create_dataloader(dataset, dataset_opt, phase)

    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch

    if opt['path']['resume_state']:
        logger.info('Loading from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    logger.info('Begin Model Evaluation.')

    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  test_data in enumerate(dataloader):
        idx += 1
        diffusion.feed_data(test_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        lr_img = tensor2img(visuals['LR'])  # uint8
        ir_img = tensor2img(visuals['IR'])  # uint8

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

        save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))


    logger.info('# Inference end#')


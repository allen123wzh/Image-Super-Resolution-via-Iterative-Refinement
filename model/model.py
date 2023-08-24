import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from torch.cuda.amp import autocast as autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

logger = logging.getLogger('base')
scaler = torch.cuda.amp.GradScaler()

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                ##########################
                ##########################
                ##########################
                # optim_params = list(self.netG.parameters())

                optim_params1 = list(self.netG.denoise_fn.parameters())
                optim_params2 = list(self.netG.global_corrector.parameters())

            # self.optG = torch.optim.Adam(
            #     optim_params, lr=opt['train']["optimizer"]["lr"])

            self.optG1 = torch.optim.Adam(
                optim_params1, lr=opt['train']["optimizer"]["lr"])
            
            self.optG2 = torch.optim.Adam(
                optim_params2, lr=opt['train']["optimizer"]["lr"])

                ##########################
                ##########################
                ##########################


            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()
        
        ### multi-gpu
        if len(opt['gpu_ids'])>1:
            assert torch.cuda.is_available()

            ### DP
            # self.netG.denoise_fn = nn.DataParallel(self.netG.denoise_fn)
            # self.netG.global_corrector = nn.DataParallel(self.netG.global_corrector)

            ### DDP
            self.netG.denoise_fn = DDP(self.netG.denoise_fn, device_ids=[opt['local_rank']])
            self.netG.global_corrector = DDP(self.netG.global_corrector, device_ids=[opt['local_rank']])

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self, it=None, grad_accum=1):
        ##########################
        ##########################
        ##########################
        
        # # for DDP grad accum efficiency
        # my_context1 = self.netG.denoise_fn.no_sync if self.opt['local_rank'] != -1 and it % grad_accum != 0 else nullcontext
        # my_context2 = self.netG.global_corrector.no_sync if self.opt['local_rank'] != -1 and it % grad_accum != 0 else nullcontext

        with autocast(dtype=torch.float16):
            # with my_context1():
            #     with my_context2():
            l_noise, l_recon = self.netG(self.data)
            # need to average in multi-gpu
            b, c, h, w = self.data['HR'].shape

            l_noise = l_noise.sum()/int(b*c*h*w)
            l_recon = l_recon.sum()/int(b*c*h*w)

            scaler.scale(l_noise).backward()
            scaler.scale(l_recon).backward()
            # scaler.scale(l_noise+l_recon).backward()

            if it % grad_accum == 0:
                # scaler.step(self.optG)
                scaler.step(self.optG1)
                scaler.step(self.optG2)
                scaler.update()
                # self.optG.zero_grad()
                self.optG1.zero_grad()
                self.optG2.zero_grad()
            
            # set log
            self.log_dict['l_noise'] = l_noise.item()
            self.log_dict['l_recon'] = l_recon.item()
        #########################
        ##########################
        ##########################

        # l_pix.backward()
        # self.optG.step()

        # set log
        # self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        # with torch.no_grad():
        #     if isinstance(self.netG, nn.DataParallel):
        #         self.SR = self.netG.module.super_resolution(
        #             self.data['SR'], continous)
        #     else:
        #         self.SR = self.netG.super_resolution(
        #             self.data['SR'], continous)
        ####################
        ####################
        ####################
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    torch.cat([self.data['SR'], self.data['hiseq']], dim=1), continous)
            
            else:
                self.SR = self.netG.super_resolution(
                    torch.cat([self.data['SR'], self.data['hiseq']], dim=1), continous)
        ####################
        ####################
        ####################
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        # gen_path = os.path.join(
        #     self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        # opt_path = os.path.join(
        #     self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # # gen
        # network = self.netG
        # if isinstance(self.netG, nn.DataParallel):
        #     network = network.module
        # state_dict = network.state_dict()
        # for key, param in state_dict.items():
        #     state_dict[key] = param.cpu()
        # torch.save(state_dict, gen_path)
        # # opt
        # opt_state = {'epoch': epoch, 'iter': iter_step,
        #              'scheduler': None, 'optimizer': None}
        # opt_state['optimizer'] = self.optG.state_dict()
        # torch.save(opt_state, opt_path)

        # logger.info(
        #     'Saved model in [{:s}] ...'.format(gen_path))
        if self.opt['local_rank'] !=0:
            return
        
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        
        if isinstance(network.denoise_fn, nn.DataParallel) or isinstance(network.denoise_fn, nn.parallel.DistributedDataParallel):
            network.denoise_fn = network.denoise_fn.module
        if isinstance(network.global_corrector, nn.DataParallel) or isinstance(network.global_corrector, nn.parallel.DistributedDataParallel):
            network.global_corrector = network.global_corrector.module
        
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None,
                     'optimizer1': None, 'optimizer2': None}
        opt_state['optimizer1'] = self.optG1.state_dict()
        opt_state['optimizer2'] = self.optG2.state_dict()

        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        # load_path = self.opt['path']['resume_state']
        # if load_path is not None:
        #     logger.info(
        #         'Loading pretrained model for G [{:s}] ...'.format(load_path))
        #     gen_path = '{}_gen.pth'.format(load_path)
        #     opt_path = '{}_opt.pth'.format(load_path)
        #     # gen
        #     network = self.netG
        #     if isinstance(self.netG, nn.DataParallel):
        #         network = network.module
        #     network.load_state_dict(torch.load(
        #         gen_path), strict=(not self.opt['model']['finetune_norm']))
        #     # network.load_state_dict(torch.load(
        #     #     gen_path), strict=False)
        #     if self.opt['phase'] == 'train':
        #         # optimizer
        #         opt = torch.load(opt_path)
        #         self.optG.load_state_dict(opt['optimizer'])
        #         self.begin_step = opt['iter']
        #         self.begin_epoch = opt['epoch']
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            # if isinstance(self.netG.denoise_fn, nn.DataParallel) or isinstance(self.netG.global_corrector, nn.DataParallel):
            #     network.denoise_fn = network.denoise_fn.module
            #     network.global_corrector = network.global_corrector.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)

                self.optG1.load_state_dict(opt['optimizer1'])
                self.optG2.load_state_dict(opt['optimizer2'])

                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

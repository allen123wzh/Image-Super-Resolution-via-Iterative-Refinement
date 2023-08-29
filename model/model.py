import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from torch.cuda.amp import autocast as autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
import copy

logger = logging.getLogger('base')
scaler = torch.cuda.amp.GradScaler()

class DDPM():
    def __init__(self, opt):
        self.opt = opt

        ### DDP
        if len(self.opt['gpu_ids'])>1:
            self.device = torch.device('cuda', self.opt['local_rank'])
        ### Single GPU
        else:
            self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

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
                optim_params1 = list(self.netG.denoise_fn.parameters())
                optim_params2 = list(self.netG.global_corrector.parameters())

            self.optG1 = torch.optim.Adam(
                optim_params1, lr=opt['train']["optimizer"]["lr"])
            
            self.optG2 = torch.optim.Adam(
                optim_params2, lr=opt['train']["optimizer"]["lr"])
            
            self.log_dict = OrderedDict()

        self.load_network(rank=self.opt['local_rank'])
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

            ### PyTorch 2.0, compile
            self.netG.denoise_fn = torch.compile(self.netG.denoise_fn)
            self.netG.global_corrector = torch.compile(self.netG.global_corrector)


    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self, it=None, grad_accum=1):
        ##########################
        ##########################
        ##########################
        
        # for DDP grad accum efficiency
        my_context1 = self.netG.denoise_fn.no_sync if self.opt['local_rank'] != -1 and it % grad_accum != 0 else nullcontext
        my_context2 = self.netG.global_corrector.no_sync if self.opt['local_rank'] != -1 and it % grad_accum != 0 else nullcontext

        with autocast(dtype=torch.float16):
            with my_context1():
                with my_context2():
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
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        
        #################
        # Need deepcopy, otherwise in DDP mode, rank 0 process's network will be different from 
        # other process, other process may wait indefinitely for rank 0's network to return to 
        # original form, DDP breaks.
        network = copy.deepcopy(self.netG)
        state_dict = network.state_dict()

        # need to add list(), otherwise ordered dict mutated
        for (key, param) in list(state_dict.items()):
            # delete the DP/DDP prefix
            if 'module.' in key:
                new_key = key.replace('module.', '')
                state_dict[new_key] = param.cpu()
                del state_dict[key]
            else:
                state_dict[key] = param.cpu()

        torch.save(state_dict, gen_path)

        # Optimizer
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None,
                    'optimizer1': None, 'optimizer2': None}
        opt_state['optimizer1'] = self.optG1.state_dict()
        opt_state['optimizer2'] = self.optG2.state_dict()

        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))


    def load_network(self, rank):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            network.load_state_dict(
                torch.load(gen_path, map_location=f'cuda:{rank}'), 
                strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)

                self.optG1.load_state_dict(opt['optimizer1'])
                self.optG2.load_state_dict(opt['optimizer2'])

                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
    

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x


    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n


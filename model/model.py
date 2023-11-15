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
from thop import profile

logger = logging.getLogger('base')
scaler = torch.cuda.amp.GradScaler()


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    

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
        self.netG = networks.define_G(opt)
        self.schedule_phase = None

        # EMA
        if opt['test']:
            self.ema_scheduler = None
        elif opt['train']['ema_scheduler']:
            self.ema_scheduler = opt['train']['ema_scheduler']
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None

        self.netG = self.set_device(self.netG)
        if self.ema_scheduler:
            self.netG_EMA = self.set_device(self.netG_EMA)
            self.netG_EMA.eval()

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule'][opt['phase']], schedule_phase=[opt['phase']])
        

        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params1 = []
                optim_params2 = list(self.netG.global_corrector.parameters())
                for k, v in self.netG.denoise_fn.named_parameters():
                    v.requires_grad = False
                    if k.find('downs.0.') >= 0:
                        v.requires_grad = True
                        # v.data.zero_()
                        optim_params1.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params1 = list(self.netG.denoise_fn.parameters())
                optim_params2 = list(self.netG.global_corrector.parameters())

            self.optG1 = torch.optim.AdamW(
                optim_params1, lr=opt['train']["optimizer"]["lr"])
            
            self.optG2 = torch.optim.AdamW(
                optim_params2, lr=opt['train']["optimizer"]["lr"])
            
            self.log_dict = OrderedDict()

        self.load_network(rank=self.opt['local_rank'] if self.opt['local_rank']!=-1 else 0)
        
        if opt['local_rank'] in {-1,0}:
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
        if opt['model']['torch_compile']:
            self.netG.denoise_fn = torch.compile(self.netG.denoise_fn)
            self.netG.global_corrector = torch.compile(self.netG.global_corrector)
            if self.ema_scheduler:
                self.netG_EMA.denoise_fn = torch.compile(self.netG_EMA.denoise_fn)
                self.netG_EMA.global_corrector = torch.compile(self.netG_EMA.global_corrector)


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

            # update EMA
            if self.ema_scheduler is not None:
                if it > self.ema_scheduler['ema_start'] and it % self.ema_scheduler['update_ema_every'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        #########################
        ##########################
        ##########################

        # l_pix.backward()
        # self.optG.step()

        # set log
        # self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                if 'IR' in self.data:
                    self.SR = self.netG.module.super_resolution(
                        torch.cat([self.data['LR'], self.data['IR']], dim=1), continous)           
                else:
                    self.SR = self.netG.module.super_resolution(self.data['LR'], continous)            
            else:
                if 'IR' in self.data:
                    self.SR = self.netG.super_resolution(
                        torch.cat([self.data['LR'], self.data['IR']], dim=1), continous)
                else:
                    self.SR = self.netG.super_resolution(self.data['LR'], continous)            
        self.netG.train()

        # Eval EMA model if exists
        if self.ema_scheduler:
            # logger.info('EMA evaluation start:')
            with torch.no_grad():
                if isinstance(self.netG_EMA, nn.DataParallel):
                    if 'IR' in self.data:
                        self.SR_EMA = self.netG_EMA.module.super_resolution(
                            torch.cat([self.data['LR'], self.data['IR']], dim=1), continous)           
                    else:
                        self.SR_EMA = self.netG_EMA.module.super_resolution(self.data['LR'], continous)            
                else:
                    if 'IR' in self.data:
                        self.SR_EMA = self.netG_EMA.super_resolution(
                            torch.cat([self.data['LR'], self.data['IR']], dim=1), continous)
                    else:
                        self.SR_EMA = self.netG_EMA.super_resolution(self.data['LR'], continous)            


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
            if self.ema_scheduler:
                if isinstance(self.netG_EMA, nn.DataParallel):
                    self.netG_EMA.module.set_new_noise_schedule(
                        schedule_opt, self.device)
                else:
                    self.netG_EMA.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach().float().cpu()
        out_dict['LR'] = self.data['LR'].detach().float().cpu()
        if 'IR' in self.data:
            out_dict['IR'] = self.data['IR'].detach().float().cpu()
        if 'HR' in self.data:
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
        if self.ema_scheduler is not None:
            out_dict['SR_EMA'] = self.SR_EMA.detach().float().cpu()
        return out_dict


    def print_network(self):
        net_desc, num_params, flops, params = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, num_params))
        logger.info(
            'UNet flops @ 256x256: {:.2f}G, with parameters: {:,d}'.format(flops/1e9, params))
        logger.info(net_desc)


    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'E{}_gen.pth'.format(epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'E{}_opt.pth'.format(epoch))
        ema_path = os.path.join(
            self.opt['path']['checkpoint'], 'E{}_ema.pth'.format(epoch))
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
            # delete torch.Compile() prefix
            elif '_orig_mod.' in key:
                new_key = key.replace('_orig_mod.', '')
                state_dict[new_key] = param.cpu()
                del state_dict[key]
            else:
                state_dict[key] = param.cpu()

        torch.save(state_dict, gen_path)
        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))
        del network
        
        ### Optimizer
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None,
                    'optimizer1': None, 'optimizer2': None}
        opt_state['optimizer1'] = self.optG1.state_dict()
        opt_state['optimizer2'] = self.optG2.state_dict()

        torch.save(opt_state, opt_path)
        logger.info(
            'Saved optimizer in [{:s}] ...'.format(opt_path))

        ### EMA
        network = copy.deepcopy(self.netG_EMA)
        state_dict = network.state_dict()

        # need to add list(), otherwise ordered dict mutated
        for (key, param) in list(state_dict.items()):
            # delete the DP/DDP prefix
            if 'module.' in key:
                new_key = key.replace('module.', '')
                state_dict[new_key] = param.cpu()
                del state_dict[key]
            # delete torch.Compile() prefix
            elif '_orig_mod.' in key:
                new_key = key.replace('_orig_mod.', '')
                state_dict[new_key] = param.cpu()
                del state_dict[key]
            else:
                state_dict[key] = param.cpu()

        torch.save(state_dict, ema_path)
        logger.info(
            'Saved EMA model in [{:s}] ...'.format(ema_path))
        del network


    def load_network(self, rank):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            ema_path = '{}_ema.pth'.format(load_path)
            
            ### Model
            network = self.netG
            state_dict = torch.load(gen_path, map_location=f'cuda:{rank}')

            for (key, param) in list(state_dict.items()):
                # delete the DP/DDP prefix
                if 'module.' in key:
                    new_key = key.replace('module.', '')
                    state_dict[new_key] = param
                    del state_dict[key]
                # delete torch.Compile() prefix
                elif '_orig_mod.' in key:
                    new_key = key.replace('_orig_mod.', '')
                    state_dict[new_key] = param
                    del state_dict[key]
                else:
                    state_dict[key] = param
                # check if stored weights match the network params' name & shape    
                if key in network.state_dict():
                    if network.state_dict()[key].shape != param.shape:
                        del state_dict[key]
                else:
                    del state_dict[key]

            network.load_state_dict(state_dict, strict=False)
            logger.info('Weights loaded.')

            ### Optimizer
            # if self.opt['phase'] == 'train':
            if os.path.exists(opt_path):
                opt = torch.load(opt_path, map_location=f'cuda:{rank}')

                self.optG1.load_state_dict(opt['optimizer1'])
                self.optG2.load_state_dict(opt['optimizer2'])

                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
                logger.info('Optimizer loaded.')
            
            ### EMA model
            if os.path.exists(ema_path):
                network = self.netG_EMA
                state_dict = torch.load(ema_path, map_location=f'cuda:{rank}')

                for (key, param) in list(state_dict.items()):
                    # delete the DP/DDP prefix
                    if 'module.' in key:
                        new_key = key.replace('module.', '')
                        state_dict[new_key] = param
                        del state_dict[key]
                    # delete torch.Compile() prefix
                    elif '_orig_mod.' in key:
                        new_key = key.replace('_orig_mod.', '')
                        state_dict[new_key] = param
                        del state_dict[key]
                    else:
                        state_dict[key] = param
                    # check if stored weights match the network params' name & shape    
                    if key in network.state_dict():
                        if network.state_dict()[key].shape != param.shape:
                            del state_dict[key]
                    else:
                        del state_dict[key]

                network.load_state_dict(state_dict, strict=False)
                logger.info('EMA model weights loaded.')
    

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


    def get_network_description(self, net):
        '''Get the string and total parameters of the network'''
        network = copy.deepcopy(net)
        if isinstance(network, nn.DataParallel):
            network = network.module
        net_desc = str(network)
        num_params = sum(map(lambda x: x.numel(), network.parameters()))

        if not self.opt['ir']:
            noisy = self.set_device(torch.randn(1,6,256,256))
        else:
            noisy = self.set_device(torch.randn(1,9,256,256))
            # noisy = self.set_device(torch.randn(1,7,256,256))
        timestep = self.set_device(torch.randint(low=0, high=1000, size=(1,)))
        macs, params = profile(network.denoise_fn, inputs=(noisy, timestep,))

        del network

        return net_desc, num_params, 2*macs, int(params)


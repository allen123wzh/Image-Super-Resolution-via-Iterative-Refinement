'''create dataset and dataloader'''
import logging
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from .LRHR_dataset import LRHRDataset


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    # mode = dataset_opt['mode']
    dataset = LRHRDataset(dataroot=dataset_opt['dataroot'],
                            split=phase,
                            data_len=dataset_opt['data_len'],
                            )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        train_sampler = None
        if dataset_opt['distributed']:
            train_sampler = DistributedSampler(dataset=dataset,
                                               num_replicas=dist.get_world_size(),
                                               rank=dist.get_rank(), 
                                               shuffle=dataset_opt['use_shuffle'])
            dataset_opt['use_shuffle'] = False
        return torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=dataset_opt['use_shuffle'],
                num_workers=dataset_opt['num_workers'],
                pin_memory=True,
                sampler=train_sampler)
    elif phase == 'val' or phase == 'test':
        return torch.utils.data.DataLoader(
                dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))

import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .ddpm import DDPM
    model = DDPM(opt)
    logger.info('Model [{:s}] is created.'.format(DDPM.__class__.__name__))
    return model

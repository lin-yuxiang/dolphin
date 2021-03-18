import torch


def get_optimizer(model, cfg):
    if hasattr(model, 'module'):
        model = model.module

    optim_cfg = cfg['optimizer']
    optim_type = optim_cfg.pop('type')
    assert optim_type in dir(torch.optim)
    optim_cfg['params'] = model.parameters()
    
    optim = getattr(torch.optim, optim_type)
    return optim(**optim_cfg)
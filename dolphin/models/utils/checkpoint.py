from collections import OrderedDict
import warnings
import pkgutil
import importlib
import torchvision
import torch
import os.path as osp
import os
import time
from torch.utils import model_zoo

from dolphin.utils.dist_utils import get_dist_info
from dolphin.utils import mkdir_or_exist, logger
from torch.nn.parallel import DataParallel, DistributedDataParallel

log = logger.get()


def load_state_dict(module, state_dict, strict=False):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, all_missing_keys, 
            unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    
    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                        f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')
    
    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            log.warning(err_msg)


def load_url_dist(url, model_dir=None):
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = importlib.import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def load_checkpoint_(filename, map_location=None):
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please '
                      'use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model, filename, map_location=None, strict=None):
    checkpoint = load_checkpoint_(filename, map_location)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict)
    else:
        load_state_dict(model, state_dict, strict)
    return checkpoint

def weights_to_cpu(state_dict):
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu

def save_checkpoint(model, filename, optimizer=None, meta=None):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(time=time.asctime())
    mkdir_or_exist(osp.dirname(filename))

    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    
    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }

    if isinstance(optimizer, torch.optim.Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()
    
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()


def check_state_dict(load_dict, new_dict):
    for k in new_dict:
        if k in load_dict:
            if new_dict[k].shape != load_dict[k].shape:
                log.debug('Skip loading parameter {}, required shape: {}, '
                        'loaded shape: {}.'.format(
                            k, load_dict[k].shape, new_dict[k].shape))
                new_dict[k] = load_dict[k]
        else:
            log.debug('Drop parameter: {}'.format(k))
    for k in load_dict:
        if not (k in new_dict):
            log.debug('No param {}.'.format(k))
            new_dict[k] = load_dict[k]
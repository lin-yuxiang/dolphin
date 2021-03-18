from torch.utils.data import DataLoader
from functools import partial
from dolphin.utils import get_dist_info
from .parallel import collate
import numpy as np
import random


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=False,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     **kwargs):
    rank, world_size = get_dist_info()
    if dist:
        pass
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
    
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=init_fn,
        shuffle=shuffle,
        **kwargs)
    
    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
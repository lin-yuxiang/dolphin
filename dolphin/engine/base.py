import torch
import copy
import time
import json
import copy
import os.path as osp
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

from dolphin import utils
from .work_flow_mixin import WorkFlowMixIn
from dolphin.models.utils import load_checkpoint, save_checkpoint
from dolphin.dataset import PCLDataParallel
from .solver import checkpointer, lr_scheduler, optimizer
from dolphin.dataset.build_dataloader import build_dataloader
from dolphin.utils import build_module_from_registers, Meters, logger


class ABCEngine(WorkFlowMixIn, metaclass=ABCMeta):

    def __init__(self,
                 algorithm=None,
                 data=None,
                 train_cfg=None,
                 test_cfg=None,
                 runtime_cfg=None,
                 meta=None,
                 distributed=None):

        kwargs = locals()
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.init_engine()
        self.set_logger()

    @property
    def epoch(self):
        return self._epoch

    @property
    def cur_iter(self):
        return self._iter
    
    @property
    def inner_iter(self):
        return self._inner_iter

    @property
    def max_epoch(self):
        return self._max_epoch

    def init_engine(self):
        self.gpu_ids = self.runtime_cfg['gpu_ids']
        self.workflow = self.runtime_cfg['work_flow']
        self.tasks = []
        for task, _ in self.workflow:
            self.tasks.append(task)
        self.meters = Meters()
        
        self.total_epochs = self.runtime_cfg['total_epochs']
        self.vis_interval = self.runtime_cfg['vis_interval']
        self.work_dir = self.runtime_cfg['work_dir']
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epoch = 0
        self._max_iter = 0

        self.model = self.build_model(
            self.algorithm, self.train_cfg, self.test_cfg)
        if 'train' in self.tasks:
            self.optimizer = self.build_optimizer(self.model, self.runtime_cfg)
            self.checkpointer = self.build_checkpointer(self.runtime_cfg)
            self.lr_scheduler = self.build_lr_scheduler(self.runtime_cfg)
        self.datasets = self.build_datasets(
            self.data, self.runtime_cfg, self.train_cfg, self.test_cfg)
        self.data_loaders = self.build_dataloaders(self.data, self.runtime_cfg)

    def set_logger(self):
        self.log = logger.get(self.runtime_cfg['log']['logger_name'])

    def init_summary(self):
        self.summary = OrderedDict()
        for i in range(self._max_epoch):
            self.summary[f'epoch_{str(i + 1)}'] = {}
    
    def build_model(self, algorithm, train_cfg, test_cfg):
        sub_cfg = dict(train_cfg=train_cfg, test_cfg=test_cfg)
        _model = build_module_from_registers(
            algorithm, module_name='algorithm', sub_cfg=sub_cfg)
        model = PCLDataParallel(
            _model.cuda(self.gpu_ids[0]), device_ids=self.gpu_ids)
        return model

    def build_datasets(self, data, runtime_cfg, train_cfg, test_cfg):
        datasets = {}
        sub_cfg = dict(train_cfg=train_cfg, test_cfg=test_cfg)
        for task in self.tasks:
            if 'test' in task:
                mode = 'test'
            else:
                mode = task
            if data.get(mode, None) is None:
                continue
            config = data[mode]
            datasets[task] = build_module_from_registers(
                config, module_name='data', sub_cfg=sub_cfg)
        return datasets

    def build_dataloaders(self, data, runtime_cfg):
        data_loaders = {}
        for task, dataset in self.datasets.items():
            if 'test' in task:
                samples_per_gpu = 1
                shuffle = False
                drop_last = False
            else:
                samples_per_gpu = data['samples_per_gpu']
                shuffle = False
                drop_last = True
            data_loaders[task] = build_dataloader(
                dataset,
                samples_per_gpu,
                data['workers_per_gpu'],
                num_gpus=len(self.gpu_ids),
                dist=self.distributed,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=runtime_cfg['seed'])
        
        return data_loaders

    def build_optimizer(self, model, runtime_cfg):
        return optimizer.get_optimizer(model, runtime_cfg)
    
    def build_lr_scheduler(self, runtime_cfg):
        lr_cfg = copy.deepcopy(runtime_cfg['lr_config'])
        if lr_cfg is None or lr_cfg == '':
            return None
        lr_scheduler_type = lr_scheduler.get_scheduler_type(lr_cfg)
        scheduler = getattr(lr_scheduler, lr_scheduler_type)
        return scheduler(**lr_cfg)
    
    def build_checkpointer(self, runtime_cfg):
        return checkpointer.Checkpointer(**runtime_cfg['checkpoint_cfg'])

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.cur_iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.cur_iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        if create_symlink:
            utils.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def cur_lr(self):
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def adjust_lr_by(self, num_batches, epoch, cur_iter, phase='epoch'):
        assert phase in ['epoch', 'iter']
        if self.lr_scheduler is None:
            return
        if isinstance(self.lr_scheduler, dict):
            if isinstance(self.optimizer, dict):
                for name, optimizer in self.optimizer.items():
                    lr_scheduler = self.lr_scheduler[name]
                    if phase == 'epoch':
                        lr_scheduler.adjust_by_epoch(
                            num_batches, optimizer, epoch, cur_iter)
                    else:
                        lr_scheduler.adjust_by_iter(epoch, cur_iter, optimizer)
        else:
            if phase == 'epoch':
                self.lr_scheduler.adjust_by_epoch(
                    num_batches, self.optimizer, epoch, cur_iter)
            else:
                self.lr_scheduler.adjust_by_iter(epoch, cur_iter, self.optimizer)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.log.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict)

    def resume(self,
               filename,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                filename,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                filename, map_location=map_location)
                
        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.log.info('resumed epoch %d, iter %d', self.epoch, self.cur_iter)
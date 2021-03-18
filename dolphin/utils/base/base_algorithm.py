from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
import torch.distributed as dist

from .base_model_module import BaseModelModule
from dolphin.utils import build_module_from_registers


class BaseAlgorithm(BaseModelModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module_cfg = locals()

    @abstractmethod
    def forward_train(self, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, **kwargs):
        pass

    def build_modules(self):
        kwargs = self.module_cfg['kwargs']
        for key, value in kwargs.items():
            module = build_module_from_registers(value, module_name=key)
            setattr(self, key, module)

    def init_weights(self, pretrained=None, pretrained_modules=None):
        kwargs = self.module_cfg['kwargs']
        pretrained = kwargs.get('pretrained', None)
        pretrained_modules = kwargs.get('pretrained_modules', None)
        if pretrained_modules is None:
            pretrained_modules = ['backbone']
        elif isinstance(pretrained_modules, str):
            pretrained_modules = [pretrained_modules]
        assert isinstance(pretrained_modules, list)
        for m in self._modules.keys():
            module = getattr(self, m)
            if m in pretrained_modules:
                module.init_weights(pretrained)
            elif 'loss' not in m:
                module.init_weights()
        # TODO IMPROVE WITH RECURSIVE

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def train_step(self, data_batch, **kwargs):
        assert isinstance(data_batch, dict)
        data_batch.update(kwargs)
        losses = self.forward(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def val_step(self, data_batch, **kwargs):
        assert isinstance(data_batch, dict)
        data_batch.update(kwargs)
        losses = self.forward(**data_batch)
        loss, log_vars = self._parse_losses(losses) 
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs
    
    def test_step(self, data_batch, **kwargs):
        assert isinstance(data_batch, dict)
        data_batch.update(kwargs)
        results = self.forward(return_loss=False, **data_batch)
        outputs = dict(results=results)
        return outputs

    @staticmethod
    def _parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.name() for _loss in loss_value)
            else:
                raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')
        
        loss = sum(
            _value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return loss, log_vars
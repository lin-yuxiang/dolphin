from itertools import chain
from torch.nn.parallel import DataParallel
from .scatter_gather import scatter_kwargs
from torch.nn.parallel import DistributedDataParallel


class PCLDataParallel(DataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
    
    def set_mode(self, mode=None):
        self.module.set_mode(mode)
    
    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.train_step(*inputs, **kwargs)
        
        assert len(self.device_ids) == 1, \
            ('PCLDataParallel only support single GPU training, if you need to'
            ' train with multiple GPUs, please use PCLDistributedDataParallel'
            ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.val_step(*inputs, **kwargs)

        assert len(self.device_ids) == 1, \
            ('PCLDataParallel only support single GPU training, if you need to'
            ' train with multiple GPUs, please use PCLDistributedDataParallel'
            ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])
    
    def test_step(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.test_step(*inputs, **kwargs)

        assert len(self.device_ids) == 1, \
            ('PCLDataParallel only support single GPU training, if you need to'
            ' train with multiple GPUs, please use PCLDistributedDataParallel'
            ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.test_step(*inputs[0], **kwargs[0])
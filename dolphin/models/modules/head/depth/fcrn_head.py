import torch
import torch.nn as nn

from dolphin.utils import Registers, build_module_from_registers, base


@Registers.head.register
class FCRNHead(base.BaseModelModule):

    def __init__(self, upsample=True, loss=None):
        super(FCRNHead, self).__init__()
        self.upsample = upsample
        self.loss_func = build_module_from_registers(loss, module_name='loss')
        if self.upsample:
            self.up_layer = nn.Upsample((228, 304), mode='bilinear')

    def forward(self, x):
        if self.upsample:
            x = self.up_layer(x)
        return x

    def init_weights(self):
        pass

    def loss(self, inputs, labels):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        
        if self.loss_func is not None:
            losses['loss_mse'] = self.loss_func(inputs, labels)
        else:
            raise ValueError(f'Please assign loss function in head module.')
        return losses
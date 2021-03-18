import torch
import torch.nn as nn

from dolphin.base.base_model_module import BaseModelModule 
from dolphin.utils import Registers, build_module_from_registers


@Registers.head.register
class FairMOTDetHead(BaseModelModule):

    def __init__(self,
                 channels,
                 first_level,
                 final_kernel=1,
                 head_conv=256,
                 num_classes=1,
                 cat_spec_wh=False,
                 reg_offset=True,
                 offset=2,
                 hm_loss=None,
                 wh_loss=None,
                 offset_loss=None,
                 **kwargs):

        super(FairMOTDetHead, self).__init__()
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.first_level = first_level

        self.hm_loss = hm_loss
        self.wh_loss = wh_loss
        self.offset_loss = offset_loss

        self.hm_channels = self.num_classes

        if not cat_spec_wh:
            self.wh_channels = 2
        else:
            self.wh_channels = 2 * self.num_classes

        if reg_offset:
            self.offset_channels = offset
        else:
            self.offset_channels = None
        
        self.heads = dict(
            hm=self.hm_channels, wh=self.wh_channels, 
            offset=self.offset_channels)
        
        for head, classes in self.heads.items():
            if self.head_conv > 0 and classes is not None:
                fc = nn.Sequential(
                    nn.Conv2d(
                        channels[self.first_level], head_conv, kernel_size=3, 
                        padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.head_conv, classes, kernel_size=final_kernel,
                        stride=1, padding=final_kernel // 2, bias=True))
            elif self.head_conv <= 0 and classes is not None:
                fc = nn.Conv2d(
                    channels[self.first_level], classes,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True)
            setattr(self, head, fc)

        self.hm_loss = build_module_from_registers(hm_loss, module_name='loss')
        self.wh_loss = build_module_from_registers(wh_loss, module_name='loss')
        self.offset_loss = build_module_from_registers(
            offset_loss,module_name='loss')

    def init_weights(self, pretrained=None):
        for head in self.heads:
            m = getattr(self, head)
            if 'hm' in head:
                if isinstance(m, nn.Sequential):
                    m[-1].bias.data.fill_(-2.19)
                else:
                    m.bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(m)
    
    def fill_fc_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = {}
        for head in self.heads:
            out[head] = getattr(self, head)(x[-1])
        return out
    
    def loss(self, inputs, label):

        if self.hm_loss_type == 'FocalLoss':
            x = inputs['hm']
            inputs['hm'] = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        
        loss_hm = self.hm_loss(inputs['hm'], label['hm'])
        if self.wh_loss['loss_weight'] > 0:
            # TODO: dense_wh
            loss_wh = self.wh_loss(
                inputs['wh'], label['reg_mask'], label['ind'], label['wh'])
        
        if self.offset_channels is not None and \
            self.offset_loss['loss_weight'] > 0:
            loss_offset = self.offset_loss(
                inputs['reg'], label['reg_mask'], label['ind'], label['reg'])
        
        losses = {
            'loss_hm': loss_hm, 'loss_wh': loss_wh, 'loss_offset': loss_offset}
        
        return losses
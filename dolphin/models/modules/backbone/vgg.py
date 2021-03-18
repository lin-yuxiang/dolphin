import torch
import torch.nn as nn

from dolphin.base.base_model_module import BaseModelModule
from dolphin.utils import Registers
from dolphin.models.utils import load_checkpoint, constant_init, kaiming_init


@Registers.backbone.register
class VGG(BaseModelModule):

    arch_settings = {
        11: (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
        13: (64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 
             512, 'M'),
        16: (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 
             'M', 512, 512, 512, 'M'),
        19: (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 
             512, 512, 'M', 512, 512, 512, 512, 'M')
    }

    def __init__(self,
                 depth, 
                 with_bn=True,
                 norm_eval=True):

        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for VGG')
        self.depth = depth
        self.with_bn = with_bn
        self.norm_eval = norm_eval
        arch = self.arch_settings[depth]
        self.make_layers(arch, batch_norm=with_bn)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
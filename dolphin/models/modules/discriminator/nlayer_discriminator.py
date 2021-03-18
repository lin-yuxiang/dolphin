import torch.nn as nn
import functools

from dolphin.utils import Registers
from dolphin.base.base_model_module import BaseModelModule
from dolphin.models.utils import Identity


@Registers.discriminator.register
class NLayerDiscriminator(BaseModelModule):
    """Defines a PatchGAN discriminator"""

    def __init__(self, 
                 in_channels, 
                 num_filters=64, 
                 n_layers=3, 
                 norm_type='batch'):
        """Construct a PatchGAN discriminator

        Parameters:
            in_channels (int)  -- the number of channels in input images
            num_filters (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if norm_type == 'batch':
            norm_layer = functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(
                nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'none':
            def norm_layer(x): return Identity()
        else:
            raise NotImplementedError

        if type(norm_layer) == functools.partial:  
            # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, num_filters, kernel_size=kw, stride=2, 
            padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  
            # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    num_filters * nf_mult_prev, num_filters * nf_mult, 
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(num_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                num_filters * nf_mult_prev, num_filters * nf_mult, 
                kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(num_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(num_filters * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  
            # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
    
    def init_weights(self):
        pass

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
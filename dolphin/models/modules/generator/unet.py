import torch
import torch.nn as nn
import functools

from dolphin.utils import Registers
from dolphin.base.base_model_module import BaseModelModule
from dolphin.models.utils import load_checkpoint, normal_init, Identity


@Registers.generator.register
class UnetGenerator(BaseModelModule):
    """Create a Unet-based generator"""

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_downs, 
                 num_filters=64,
                 init_gain=None,
                 norm_type='batch', 
                 use_dropout=False):
        """Construct a Unet generator
        Parameters:
            in_channels (int)  -- the number of channels in input images
            out_channels (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. 
                               For example, # if |num_downs| == 7,
                               image of size 128x128 will become of size 1x1 
                               # at the bottleneck
            num_filters (int)  -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
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

        self.init_gain = init_gain
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            num_filters * 8, num_filters * 8, input_nc=None, submodule=None, 
            norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for _ in range(num_downs - 5):          
            # add intermediate layers with num_filter * 8 filters
            unet_block = UnetSkipConnectionBlock(
                num_filters * 8, num_filters * 8, input_nc=None, 
                submodule=unet_block, norm_layer=norm_layer, 
                use_dropout=use_dropout)
        # gradually reduce the number of filters from num_filters * 8 
        # to num_filters 
        unet_block = UnetSkipConnectionBlock(
            num_filters * 4, num_filters * 8, input_nc=None, 
            submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            num_filters * 2, num_filters * 4, input_nc=None, 
            submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            num_filters , num_filters * 2, input_nc=None, 
            submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            out_channels, num_filters, input_nc=in_channels,
            submodule=unet_block, outermost=True, norm_layer=norm_layer)  
            # add the outermost layer

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=self.init_gain)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    normal_init(m, mean=1, std=self.init_gain)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, 
                 outer_nc, 
                 inner_nc, 
                 input_nc=None,
                 submodule=None, 
                 outermost=False, 
                 innermost=False, 
                 norm_layer=nn.BatchNorm2d, 
                 use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
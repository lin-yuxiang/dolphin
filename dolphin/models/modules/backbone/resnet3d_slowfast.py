import torch.nn as nn
import torch
import warnings

from dolphin.utils import Registers, build_module_from_registers
from dolphin.models.utils import kaiming_init, load_checkpoint, load_checkpoint_
from .resnet3d import ResNet3d


@Registers.backbone.register
class ResNet3dPathway(ResNet3d):

    def __init__(self,
                 *args,
                 lateral=False,
                 speed_ratio=8,
                 channel_ratio=8,
                 fusion_kernel=5,
                 **kwargs):
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)
        self.inplanes = self.base_channels
        if self.lateral:
            self.conv1_lateral = nn.Conv3d(
                self.inplanes // self.channel_ratio,
                self.inplanes * 2 // self.channel_ratio,
                kernel_size=(fusion_kernel, 1, 1),
                stride=(self.speed_ratio, 1, 1),
                padding=((fusion_kernel - 1) // 2, 0, 0),
                bias=False)

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2 ** i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1:
                # no lateral connection in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                setattr(
                    self, lateral_name,
                    nn.Conv3d(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * 2 // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False))
                self.lateral_connections.append(lateral_name)

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       num_blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       inflate=1,
                       inflate_style='3x1x1',
                       with_checkpoint=False):
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * num_blocks
        assert len(inflate) == num_blocks
        if self.lateral:
            lateral_inplanes = inplanes * 2 // self.channel_ratio
        else:
            lateral_inplanes = 0
        if (spatial_stride != 1
                or (inplanes + lateral_inplanes) != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv3d(
                    inplanes + lateral_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(temporal_stride, spatial_stride, spatial_stride),
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion))
        else:
            downsample = None
        
        layers = []
        layers.append(
            block(
                inplanes + lateral_inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                with_checkpoint=with_checkpoint))

        inplanes = planes * block.expansion

        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    1,
                    1,
                    dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    with_checkpoint=with_checkpoint))

        return nn.Sequential(*layers)

    def inflate_weights(self):
        state_dict_r2d = load_checkpoint_(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']
        
        inflated_param_names = []
        for name, module in self.named_modules():
            if 'lateral' in name:
                continue
            elif 'conv' in name:
                if name + '.weight' not in state_dict_r2d:
                    warnings.warn(f'Module not exist in the state_dict_r2d'
                                   f': {name}')
                else:
                    shape_2d = state_dict_r2d[name + '.weight'].shape
                    shape_3d = module.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        warnings.warn(
                            f'Weight shape mismatch for '
                            f': {name} : '
                            f'3d weight shape: {shape_3d}; '
                            f'2d weight shape: {shape_2d}. ')
                    else:
                        self._inflate_conv_params(
                            module, state_dict_r2d, name, inflated_param_names)
            elif 'bn' in name:
                if name + '.weight' not in state_dict_r2d:
                    warnings.warn(
                        f'Module not exist in the state_dict_r2d: {name}')
                else:
                    self._inflate_bn_params(
                        module, state_dict_r2d, name, inflated_param_names)
        
        remaining_names = set(
            state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            warnings.warn(
                f'These parameters in the 2d checkpoint are not loaded'
                f': {remaining_names}')

    def _inflate_conv_params(self, conv3d, state_dict_r2d, module_name_2d,
                             inflated_param_names):
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_r2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]
        if new_shape[1] != old_shape[1]:
            new_channels = new_shape[1] - old_shape[1]
            # pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels, ) + pad_shape[2:]
            conv2d_weight = torch.cat(
                (conv2d_weight,
                 torch.zeros(pad_shape).type_as(conv2d_weight).to(
                     conv2d_weight.device)),
                dim=1)
        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_r2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            
            if (i != len(self.res_layers) and self.lateral):
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False
    
    def init_weights(self, pretrained=None):
        super().init_weights(pretrained=pretrained)
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    kaiming_init(m)


@Registers.backbone.register
class ResNet3dSlowFast(nn.Module):

    def __init__(self,
                 pretrained,
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 slow_pathway=dict(
                     type='ResNet3dPathway',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     dilations=(1, 1, 1, 1),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 fast_pathway=dict(
                     type='ResNet3dPathway',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1)):
        super().__init__()
        self.pretrained = pretrained
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if slow_pathway['lateral']:
            slow_pathway['speed_ratio'] = speed_ratio
            slow_pathway['channel_ratio'] = channel_ratio

        self.slow_path = build_module_from_registers(
            slow_pathway, module_name='backbone')
        self.fast_path = build_module_from_registers(
            fast_pathway, module_name='backbone')

    def init_weights(self):
        if isinstance(self.pretrained, str):
            load_checkpoint(self, self.pretrained)
        elif self.pretrained is None:
            self.fast_path.init_weights(self.pretrained)
            self.slow_path.init_weights(self.pretrained)
        else:
            raise TypeError('pretrained must be a str of None')
    
    def forward(self, x):
        x_slow = x[:, :, ::self.resample_rate, :, :]
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.bn1(x_slow)
        x_slow = self.slow_path.relu(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = x[:, :, ::self.resample_rate // self.speed_ratio, :, :]
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.bn1(x_fast)
        x_fast = self.fast_path.relu(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)
            if (i != len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral):
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        out = (x_slow, x_fast)

        return out
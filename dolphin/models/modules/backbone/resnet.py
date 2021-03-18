import torch.nn as nn
import torch.utils.checkpoint as cp

from dolphin.models.utils import kaiming_init, constant_init, load_checkpoint
from dolphin.utils import Registers
from dolphin.base.base_model_module import BaseModelModule


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None, 
                 groups=1,
                 base_width=64, 
                 dilation=1, 
                 norm_layer=None, 
                 with_checkpoint=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'syncbatch':
            norm_layer = nn.SyncBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.with_checkpoint = with_checkpoint

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out) 
 
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out
        
        if self.with_checkpoint and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
            
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None, 
                 groups=1,
                 base_width=64, 
                 dilation=1, 
                 norm_layer=None, 
                 with_checkpoint=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.with_checkpoint = with_checkpoint

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out
        
        if self.with_checkpoint and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResLayer(nn.Sequential):
    def __init__(self, 
                 block, 
                 inplanes, 
                 planes, 
                 num_blocks, 
                 stride=1, 
                 avg_down=False, 
                 groups=1, 
                 base_width=64,
                 dilation=1, 
                 norm_layer=None, 
                 contract_dilation=False,
                 with_checkpoint=False):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))

            downsample.extend([
                nn.Conv2d(
                    inplanes, planes * block.expansion, kernel_size=1, 
                    stride=conv_stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            ])
            downsample = nn.Sequential(*downsample)
        
        layers = []
        if dilation > 1 and contract_dilation:
            first_dilation = dilation // 2
        else:
            first_dilation = dilation
        layers.append(
            block(inplanes=inplanes, 
                  planes=planes, 
                  stride=stride, 
                  downsample=downsample,
                  dilation=first_dilation,
                  with_checkpoint=with_checkpoint))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(inplanes=inplanes, 
                      planes=planes, 
                      stride=1,
                      dilation=dilation,
                      with_checkpoint=with_checkpoint)
            )
        super(ResLayer, self).__init__(*layers)


@Registers.backbone.register
class ResNet(BaseModelModule):
    __factory = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, 
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=[3],
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 with_checkpoint=False,
                 zero_init_residual=True,
                 contract_dilation=False,
                 norm_layer=None,
                 norm_eval=True):

        super(ResNet, self).__init__()
        if depth not in self.__factory:
            raise KeyError(f'invalid depth {depth} for resnet')

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert self.num_stages >= 1 and self.num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.with_checkpoint = with_checkpoint
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.__factory[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.norm_eval = norm_eval

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for index, num_blocks in enumerate(self.stage_blocks):
            stride = strides[index]
            dilation = dilations[index]

            planes = base_channels * 2 ** index
            res_layer = ResLayer(
                block=self.block, 
                inplanes=self.inplanes, 
                planes=planes,
                num_blocks=num_blocks, 
                stride=stride, 
                dilation=dilation,
                contract_dilation=contract_dilation,
                avg_down=self.avg_down, 
                with_checkpoint=self.with_checkpoint)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{index + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (
            len(self.stage_blocks) - 1)


    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                conv3x3(in_channels, stem_channels // 2, stride=2),
                self._norm_layer(stem_channels // 2),
                nn.ReLU(inplace=True),
                conv3x3(stem_channels // 2, stem_channels // 2, stride=1),
                self._norm_layer(stem_channels // 2),
                nn.ReLU(inplace=True),
                conv3x3(stem_channels // 2, stem_channels, stride=1),
                self._norm_layer(stem_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, stem_channels, kernel_size=7, stride=2, 
                padding=3, bias=False)
            self.bn1 = self._norm_layer(stem_channels)
            self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.bn1.eval()
                for m in [self.conv1, self.bn1]:
                    for param in m.parameters():
                        param.requires_grad = False
        
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
            
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.bn3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.bn2, 0)
        
        else:
            raise TypeError('pretrained must be a str or None')

    
    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


@Registers.backbone.register
class ResNetV1c(ResNet):
    
    def __init__(self, **kwargs):
        super(ResNetV1c, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


@Registers.backbone.register
class ResNetV1d(ResNet):

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
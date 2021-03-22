import torch
import torch.nn as nn
import torch.nn.functional as F

from dolphin.utils import Registers, base
from dolphin.models.utils import normal_init
from dolphin.loss import accuracy


@Registers.head.register
class FCNHead(base.BaseModelModule):

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 dropout_ratio=0.1,
                 in_index=-1,
                 input_transform=None,
                 loss=dict(),
                 ignore_index=255,
                 align_corners=False,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        super(FCNHead, self).__init__()

        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        assert num_convs > 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        loss_type = loss.pop('type')
        self.loss_seg = Registers.loss[loss_type](**loss)

        convs = []
        convs.extend([
            nn.Conv2d(self.in_channels, self.channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)])
        for _ in range(num_convs - 1):
            convs.extend([
                nn.Conv2d(self.channels, self.channels, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True)])
        self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(self.in_channels + self.channels, self.channels,
                          kernel_size=kernel_size, padding=kernel_size // 2,
                          bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True))

    def init_weights(self, pretrained=None):
        normal_init(self.conv_seg, mean=0, std=0.01)
    
    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [F.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        return output

    def loss(self, seg_logit, seg_label):
        losses = dict()
        seg_logit = F.interpolate(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        losses['loss_seg'] = self.loss_seg(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        losses['acc_seg'] = accuracy(seg_logit, seg_label)
        return losses
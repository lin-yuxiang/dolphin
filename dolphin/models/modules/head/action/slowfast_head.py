import torch
import torch.nn as nn

from dolphin.models.utils import normal_init
from dolphin.utils import Registers, build_module_from_registers
from dolphin.base.base_model_module import BaseModelModule
from dolphin.loss import top_k_accuracy


@Registers.head.register
class SlowFastHead(BaseModelModule):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01):
        super().__init__()
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None
        
        self.loss_cls = build_module_from_registers(
            loss_cls, module_name='loss')

    def init_weights(self, pretrained=None):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # N, channel_fast, T, H, W   N, channel_slow, T, H, W
        x_fast, x_slow = x
        # N, channel_fast, 1, 1, 1
        x_fast = self.avg_pool(x_fast)
        # N, channel_slow, 1, 1, 1
        x_slow = self.avg_pool(x_slow)
        # N, channel_fast + channel_slow, 1, 1, 1
        x = torch.cat((x_slow, x_fast), dim=1)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        
        return cls_score

    def loss(self, cls_score, labels):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        if not self.multi_class:
            top_k_acc = top_k_accuracy(
                cls_score.detach().cpu().numpy(),
                labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tesnor(
                top_k_acc[1], device=cls_score.device)

        elif self.label_smooth_eps != 0:
            labels = (
                (1 - self.label_smooth_eps) * labels + self.label_smooth_eps / 
                 self.num_classes)

        losses['loss_cls'] = self.loss_cls(cls_score, labels)
        return losses
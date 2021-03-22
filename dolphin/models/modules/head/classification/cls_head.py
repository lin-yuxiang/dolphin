import torch.nn as nn

from dolphin.models.utils import normal_init
from dolphin.utils import Registers, base


@Registers.head.register
class ClassifierHead(base.BaseModelModule):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 with_sigmoid=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.with_sigmoid = with_sigmoid
        if with_sigmoid:
            self.sigmoid = nn.Sigmoid()
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self, pretrained=None):
        normal_init(self.fc, std=self.init_std)

    def forward(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        cls_score = self.fc(x)
        if self.with_sigmoid:
            cls_score = self.sigmoid(cls_score)
        return cls_score
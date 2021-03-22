import torch
import torch.nn as nn

from dolphin.models.utils import normal_init
from dolphin.utils import Registers, base


@Registers.head.register
class ClassifierHead2d(base.BaseModelModule):

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
        # self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 2),
            nn.Sigmoid(),
            nn.Linear(self.in_channels // 2, self.num_classes))

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

    def init_weights(self, pretrained=None):
        if isinstance(self.fc, nn.Sequential):
            for layer in self.fc:
                if not hasattr(layer, 'weight'):
                    continue
                normal_init(layer, std=self.init_std)
        else:
            normal_init(self.fc, std=self.init_std)
    
    def forward(self, x, img_meta):
        x = x[0]
        x = self.avg_pool(x)
        x = x.squeeze(-1).view(-1)
        # x = x.squeeze(-1).squeeze(-1)
        if x.size(0) != self.in_channels:
            import pdb; pdb.set_trace()
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc(x)
        cls_score = self.sigmoid(cls_score)
        return cls_score
    
#     def forward(self, x, gt_bbox, img_meta):
#         assert isinstance(x, (tuple, list))
#         feat_boxs = []
#         for feats in x:
#             feat_box = []
#             for feat, (cat, boxs) in zip(feats, gt_bbox.items()):
#                 _, h, w = feat.size()
#                 for box in boxs:
#                     xmin = int(box['xmin'] * w + 0.5)
#                     ymin = int(box['ymin'] * h + 0.5)
#                     box_w = int((box['xmax'] - box['xmin']) * w) + 1
#                     box_h = int((box['ymax'] - box['ymin']) * h) + 1
#                     box_feat = feat[:, ymin: ymin + box_h, xmin: xmin + box_w]
#                     box_feat = self.avg_pool(box_feat).squeeze()
#                     feat_box.append(box_feat)
                
#                 box_num = img_meta[0]['box_num']
#                 if len(boxs) < box_num[cat]:
#                     box_feat = feat_box[-1]
#                     for times in range(box_num[cat] - len(boxs)):
#                         feat_box.append(torch.zeros_like(box_feat))
                
#             feat_box = torch.cat(feat_box)
#             feat_boxs.append(feat_box)
#         feat_boxs = torch.cat(feat_boxs)
#         if feat_boxs.size(0) != self.in_channels:
#             import pdb; pdb.set_trace()

#         if self.dropout is not None:
#             feat_boxs = self.dropout(feat_boxs)
#         cls_score = self.fc(feat_boxs)
#         cls_score = self.sigmoid(cls_score)
#         return cls_score

    def loss(self, cls_score, label):
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)

        losses['loss_cls'] = self.loss_cls(cls_score, label)
        return losses
import torch
import torch.nn as nn

from dolphin.models.utils import normal_init
from dolphin.utils import Registers
from dolphin.base.base_model_module import BaseModelModule


@Registers.head.register
class ClassifierHeadBaseline(BaseModelModule):

    def __init__(self,
                 num_classes,
                 in_channels,
                 fc_channels=512,
                 loss_cls=dict(),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 num_feats=1,
                 with_sigmoid=False,
                 with_softmax=False,
                 with_mask=False,
                 with_3d_conv=False,
                 with_hidden_layer=False,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.with_sigmoid = with_sigmoid
        self.with_mask = with_mask
        self.with_3d_conv = with_3d_conv
        self.with_hidden_layer=with_hidden_layer
        if with_sigmoid:
            self.sigmoid = nn.Sigmoid()
        if with_softmax:
            self.softmax = nn.Softmax()
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        if self.with_hidden_layer:
            self.fc = nn.Sequential(
                nn.Linear(fc_channels, fc_channels // 2),
                nn.Sigmoid(),
                nn.Linear(fc_channels // 2, self.num_classes))
        else:
            self.fc = nn.Linear(fc_channels, self.num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None
            
        if self.with_3d_conv:
            self.conv_last = nn.Conv3d(
                self.in_channels,
                fc_channels,
                kernel_size=(14 * num_feats, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False)

    def init_weights(self, pretrained=None):
        if isinstance(self.fc, nn.Sequential):
            for layer in self.fc:
                if not hasattr(layer, 'weight'):
                    continue
                normal_init(layer, std=self.init_std)
        else:
            normal_init(self.fc, std=self.init_std)
            
    def mask_out(self, x, gt_bbox, img_meta):
        mask = torch.zeros_like(x)
        feat_w = x.size(-1)
        feat_h = x.size(-2)
        for idx, filename in enumerate(img_meta[0]['filename']):
            cat = filename.split('.')[0].split('/')[-1]
            boxes = gt_bbox[cat]
            for box in boxes:
#                 xmin = int(box['xmin'] * feat_w + 0.5)
#                 ymin = int(box['ymin'] * feat_h + 0.5)
#                 w = int((box['xmax'] - box['xmin']) * feat_w) + 1
#                 h = int((box['ymax'] - box['ymin']) * feat_h) + 1
#                 mask[idx, :, ymin: ymin + h, xmin: xmin + w] = 1
                xmin = int(box['xmin'] * feat_w)
                ymin = int(box['ymin'] * feat_h)
                xmax = int(box['xmax'] * feat_w + 0.5)
                ymax = int(box['ymax'] * feat_h + 0.5)
                mask[idx, :, ymin: ymax + 1, xmin: xmax + 1] = 1
        out = mask * x
        return out
    
    def forward(self, x, gt_bbox, img_meta):
        feats = []
        for feat in x:
            if self.with_mask:
                feat = self.mask_out(feat, gt_bbox, img_meta)
            feat = self.avg_pool(feat)
            feats.append(feat)
        feats = torch.cat(feats)
        if self.with_3d_conv:
            feats = self.conv_last(feats.unsqueeze(0).transpose(1, 2))
        feats = feats.view(-1)
        # x = x.squeeze(-1).squeeze(-1)
        if self.dropout is not None:
            feats = self.dropout(feats)
        cls_score = self.fc(feats)
        cls_score = self.sigmoid(cls_score)
        return cls_score

    def loss(self, cls_score, label):
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)

        losses['loss_cls'] = self.loss_cls(cls_score, label)
        return losses
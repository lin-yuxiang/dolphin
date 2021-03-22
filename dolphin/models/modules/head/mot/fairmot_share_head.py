import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from dolphin.utils import Registers, base


@Registers.head.register
class FairMOTShareHead(base.BaseModelModule):

    def __init__(self,
                 channels=None,
                 down_ratio=None,
                 final_kernel=1,
                 head_conv=256,
                 num_classes=1,
                 det_head=dict(
                     typd='DetHead',
                     cat_spec_wh=False,
                     reg_offset=True,
                     offset=2,
                     hm_loss=dict(
                         type='FocalLoss', loss_weight=1.0),
                     wh_loss=dict(
                         type='RegL1Loss', loss_weight=0.1),
                     offset_loss=dict(
                         type='RegL1Loss', loss_weight=1.0)),
                 id_head=dict(
                     type='ReIDHead',
                     reid_dim=512,
                     reid_nID=14455,
                     id_loss=dict(
                         type='CrossEntropyLoss', loss_weight=1.0, 
                         ignore_index=-1)),
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(FairMOTShareHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        first_level = int(np.log2(down_ratio))

        if det_head is not None:
            det_head.update(dict(
                    num_classes=num_classes, channels=channels, 
                    first_level=first_level, final_kernel=final_kernel, 
                    head_conv=head_conv))
            det_head_type = det_head.pop('type')
            self.det_head = Registers.head[det_head_type](**det_head)
        else:
            self.det_head = None

        if id_head is not None:
            id_head.update(dict(
                channels=channels, first_level=first_level,
                final_kernel=final_kernel, head_conv=head_conv))
            id_head_type = id_head.pop('type')
            self.id_head = Registers.head[id_head_type](**id_head)
        else:
            self.id_head = None

        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def init_weights(self, pretrained=None):
        if self.det_head is not None:
            self.det_head.init_weights(pretrained)
        if self.id_head is not None:
            self.id_head.init_weights(pretrained)
    
    def forward(self, x, label):
        out = {}
        if self.det_head is not None: 
            det_out = self.det_head(x)
            out.update(det_out)
        if self.id_head is not None:
            id_out = self.id_head(x, label)
            out.update(id_out)
        return out
    
    def loss(self, inputs, label):
        losses = self.det_head.loss(inputs, label)
        id_loss = self.id_head.loss(inputs['id'], label)
        losses.update(id_loss)

        for loss, loss_value in losses.items():
            if 'id' in loss:
                losses[loss] = loss_value * 0.5 * torch.exp(-self.s_id)
            else:
                losses[loss] = loss_value * 0.5 * torch.exp(-self.s_det)
        
        losses['loss_s'] = 0.5 * (self.s_det + self.s_id)
        
        return losses
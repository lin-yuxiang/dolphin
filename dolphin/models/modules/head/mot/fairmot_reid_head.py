import torch.nn as nn
import torch.nn.functional as F
import math

from dolphin.base.base_model_module import BaseModelModule 
from dolphin.utils import (Registers, tranpose_and_gather_feat, 
                         build_module_from_registers)


@Registers.head.register
class FairMOTReIDHead(BaseModelModule):

    def __init__(self,
                 channels,
                 first_level,
                 final_kernel=1,
                 head_conv=256,
                 reid_dim=512,
                 reid_nID=14455,
                 id_loss=None,
                 **kwargs):

        super(FairMOTReIDHead, self).__init__()
        self.head_conv = head_conv
        self.first_level = first_level

        self.reid_nID = reid_nID
        self.embedding_dim = reid_dim
        self.embedding_scale = math.sqrt(2) * math.log(self.reid_nID - 1)
        
        if self.head_conv > 0 and self.embedding_dim is not None:
            self.id = nn.Sequential(
                nn.Conv2d(
                    channels[self.first_level], head_conv, kernel_size=3, 
                    padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.head_conv, self.embedding_dim, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
        elif self.head_conv <= 0 and self.embedding_dim is not None:
            self.id = nn.Conv2d(
                channels[self.first_level], self.embedding_dim,
                kernel_size=final_kernel, stride=1,
                padding=final_kernel // 2, bias=True)

        self.id_loss = build_module_from_registers(id_loss, module_name='loss')
        self.id_classifier = nn.Linear(self.embedding_dim, self.reid_nID)

    def init_weights(self, pretrained=None):
        for m in self.id.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_feat(self, x, label):
        id_head = tranpose_and_gather_feat(x, label['ind'])
        id_head = id_head[label['reg_mask'] > 0].contiguous()
        id_head = self.embedding_scale * F.normalize(id_head)
        return id_head
    
    def forward(self, x, label):
        out = self.id(x[-1])
        if self.mode != 'test_track':  # TODO notice
            out = self.extract_feat(out, label)
        return {'id': out}
    
    def loss(self, inputs, label):
        inputs = self.id_classifier(inputs).contiguous()
        id_target = label['ids'][label['reg_mask'] > 0]
        loss_id = self.id_Loss(inputs, id_target)
        
        losses = {'loss_id': loss_id}
        
        return losses
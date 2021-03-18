import torch
import torch.nn as nn

from dolphin.base.base_model_module import BaseModelModule 
from dolphin.models.utils import normal_init
from dolphin.utils import Registers, build_module_from_registers


@Registers.head.register
class ActivateLearningHead(BaseModelModule):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        
        super(ActivateLearningHead, self).__init__()
        
        self.init_std = init_std
        self.in_channels = in_channels
        self.loss_func = build_module_from_registers(loss, 'loss')
        self.drop = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_channels, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def init_weights(self, pretrained=None):
        normal_init(self.fc1, std=self.init_std)
        normal_init(self.fc2, std=self.init_std)

    def forward(self, x):
        x = x.view(-1, self.in_channels)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
    def loss(self, inputs, labels):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        
        if self.loss_func is not None:
            losses['loss_ce'] = self.loss_func(inputs, labels)
        else:
            raise ValueError(f'Please assign loss function in head module.')
        return losses

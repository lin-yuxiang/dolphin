import torch.nn as nn

from dolphin.base.base_model_module import BaseModelModule
from dolphin.models.utils import load_checkpoint, kaiming_init, constant_init
from dolphin.utils import Registers


@Registers.backbone.register
class ActivateLearningBackbone(BaseModelModule):

    def __init__(self, in_channels=1):
        super(ActivateLearningBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2)
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise ValueError(f'pretrained should be str or None.')
    
    def forward(self, imgs):
        x = self.conv1(imgs)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.pool2(x)
        x = self.relu(x)
        return x
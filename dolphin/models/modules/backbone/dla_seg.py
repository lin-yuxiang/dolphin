import numpy as np

from dolphin.utils import Registers
from .dla import DLA, DLAUp, IDAUp, BasicBlock
from dolphin.base.base_model_module import BaseModelModule
from dolphin.models.utils import load_checkpoint


@Registers.backbone.register
class DLASeg(BaseModelModule):
    
    def __init__(self, 
                 depth,
                 levels=[1, 1, 1, 2, 2, 1],
                 channels=[16, 32, 64, 128, 256, 512],
                 down_ratio=4,
                 last_level=5,
                 out_channels=0,
                 **kwargs):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = DLA(levels, channels, block=BasicBlock)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales)
        
        if out_channels == 0:
            out_channels = channels[self.first_level]

        up_f = self.last_level - self.first_level
        self.ida_up = IDAUp(
            out_channels, channels[self.first_level: self.last_level],
            [2 ** i for i in range(up_f)])

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is not None:
            raise TypeError('pretrained must be str or None.')

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y
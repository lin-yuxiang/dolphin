import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from dolphin.utils import Registers
from dolphin.base.base_model_module import BaseModelModule
from dolphin.models.utils import load_checkpoint, normal_init, constant_init


class UpProject(nn.Module):

    def __init__(self, in_channels, out_channels, pad_mode):
        super(UpProject, self).__init__()

        assert pad_mode in ['left', 'right']
        self.pad_mode = pad_mode

        self.pad = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0]]
        if self.pad_mode == 'right':
            for idx, pad in enumerate(self.pad):
                if pad[0] + pad[1] == 1:
                    pad[0], pad[1] = pad[1], pad[0]
                if pad[2] + pad[3] == 1:
                    pad[2], pad[3] = pad[3], pad[2]

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, 2)
        
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batch_size = x.size(0)
        out1_1 = self.conv1_1(F.pad(x, self.pad[0]))
        out1_2 = self.conv1_2(F.pad(x, self.pad[1]))
        out1_3 = self.conv1_3(F.pad(x, self.pad[2]))
        out1_4 = self.conv1_4(F.pad(x, self.pad[3]))

        out2_1 = self.conv2_1(F.pad(x, self.pad[0]))
        out2_2 = self.conv2_2(F.pad(x, self.pad[1]))
        out2_3 = self.conv2_3(F.pad(x, self.pad[2]))
        out2_4 = self.conv2_4(F.pad(x, self.pad[3]))

        h, w = out1_1.size(2), out1_1.size(3)

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(
            0, 1, 3, 4, 2).contiguous().view(batch_size, -1, h, w * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(
            0, 1, 3, 4, 2).contiguous().view(batch_size, -1, h, w * 2)
        
        out1 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(
            0, 1, 3, 2, 4).contiguous().view(batch_size, -1, h * 2, w * 2)
        
        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(
            0, 1, 3, 4, 2).contiguous().view(batch_size, -1, h, w * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(
            0, 1, 3, 4, 2).contiguous().view(batch_size, -1, h, w * 2) 
        
        out2 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(
            0, 1, 3, 2, 4).contiguous().view(batch_size, -1, h * 2, w * 2) 
        
        out1 = self.bn1_1(out1)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2)

        out = out1 + out2
        out = self.relu(out)

        return out


@Registers.decoder.register
class FCRNUpProjection(BaseModelModule):

    def __init__(self, pad_mode='left'):
        super(FCRNUpProjection, self).__init__()
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)

        self.up1 = self._make_upproj_layer(UpProject, 1024, 512, pad_mode)
        self.up2 = self._make_upproj_layer(UpProject, 512, 256, pad_mode)
        self.up3 = self._make_upproj_layer(UpProject, 256, 128, pad_mode)
        self.up4 = self._make_upproj_layer(UpProject, 128, 64, pad_mode)

        self.drop = nn.Dropout2d()

        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_upproj_layer(self, block, in_channels, out_channels, pad_mode):
        return block(in_channels, out_channels, pad_mode)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    normal_init(m, mean=0, std=math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be str or None.')
    
    def forward(self, x):
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.drop(x)

        x = self.conv3(x)
        x = self.relu(x)

        return x
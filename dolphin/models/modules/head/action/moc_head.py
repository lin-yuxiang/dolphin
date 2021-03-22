import torch
import torch.nn as nn

from dolphin.utils.postprocess import moc_decode, moc_post_process
from dolphin.utils import Registers, build_module_from_registers, base


@Registers.head.register
class MOCHead(base.BaseModelModule):

    def __init__(self,
                 in_channels=0,
                 head_conv=0,
                 num_classes=0,
                 output_stride=4,
                 K=7,
                 max_output_objects=0,
                 hm_loss=None,
                 mov_loss=None,
                 wh_loss=None):
        
        super(MOCHead, self).__init__()
        assert head_conv > 0

        head_info = dict(hm=num_classes, mov=2 * K, wh=2 * K)

        self.hm = nn.Sequential(
            nn.Conv2d(
                K * in_channels, head_conv, kernel_size=3, 
                padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, head_info['hm'], kernel_size=1, stride=1,
                padding=0, bias=True))
        
        self.mov = nn.Sequential(
            nn.Conv2d(
                K * in_channels, head_conv, kernel_size=3, padding=1, 
                bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, head_info['mov'], kernel_size=1, stride=1,
                padding=0, bias=True))

        self.wh = nn.Sequential(
            nn.Conv2d(
                in_channels, head_conv, kernel_size=3, padding=1, 
                bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, head_info['wh'] // K, kernel_size=1, stride=1,
                padding=0, bias=True))

        self.hm_loss = build_module_from_registers(hm_loss, module_name='loss')
        self.wh_loss = build_module_from_registers(wh_loss, module_name='loss')
        self.mov_loss = build_module_from_registers(
            mov_loss, module_name='loss')
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.N = max_output_objects
        self.K = K

    def init_weight(self, pretrained=None):
        self.fill_fc_weights(self.hm)
        self.fill_fc_weights(self.mov)
        self.fill_fc_weights(self.wh)

    def fill_fc_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = {}
        output_wh = []
        for feat in x:
            output_wh.append(self.wh(feat))
        x = torch.cat(x, dim=1)
        output_wh = torch.cat(output_wh, dim=1)
        output['hm'] = torch.clamp(self.hm(x).sigmoid_(), min=1e-4, max=1-1e-4)
        output['mov'] = self.mov(x)
        output['wh'] = output_wh
        return output

    def loss(self, x, label):
        hm_loss = self.hm_loss(x['hm'], label['hm'])
        mov_loss = self.mov_loss(
            x['mov'], label['mask'], label['index'], label['mov'])
        wh_loss = self.wh_loss(
            x['wh'], label['mask'], label['index'], label['wh'], 
            index_all=label['index_all'])
        losses = dict(loss_hm=hm_loss, loss_mov=mov_loss, loss_wh=wh_loss)
        return losses

    def process(self, x, video_meta):
        hm = x['hm']
        mov = x['mov']
        wh = x['wh']
        h, w = video_meta['imgs_cfg']['original_shape']
        new_h, new_w = video_meta['imgs_cfg']['shape']
        out_h, out_w = new_h // self.output_stride, new_w // self.output_stride
        detections = moc_decode(hm, wh, mov, N=self.N, K=self.K)
        detections = moc_post_process(
            detections, h, w, out_h, out_w, self.num_classes, self.K)
        return detections
import math
import numpy as np
import torch
import torch.nn as nn

from dolphin.utils import Registers, build_module_from_registers
from dolphin.base.base_model_module import BaseModelModule
from dolphin.models.utils import kaiming_init


@Registers.head.register
class BMNHead(BaseHead):

    def __init__(self,
                 temporal_scale,
                 prop_boundary_ratio,
                 num_samples,
                 num_samples_per_bin,
                 feat_dim,
                 hidden_dim_1d=256,
                 hidden_dim_2d=128,
                 hidden_dim_3d=512,
                 tem_loss=dict(),
                 pem_reg_loss=dict(),
                 pem_cls_loss=dict(),
                 **kwargs):
        super(BMNHead, self).__init__()

        self.tscale = temporal_scale
        self.temporal_gap = 1. / temporal_scale
        self.pbr = prop_boundary_ratio
        self.num_samples = num_samples
        self.num_samples_per_bin = num_samples_per_bin
        self.feat_dim = feat_dim
        self.hidden_dim_1d = hidden_dim_1d
        self.hidden_dim_2d = hidden_dim_2d
        self.hidden_dim_3d = hidden_dim_3d

        self.tem_loss = build_module_from_registers(
            tem_loss, module_name='loss')
        self.pem_reg_loss = build_module_from_registers(
            pem_reg_loss, module_name='loss')
        self.pem_cls_loss = build_module_from_registers(
            pem_cls_loss, module_name='loss')

        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3,
                    padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3,
                    padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3,
                    padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3,
                    padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()    
        ) 

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_2d, kernel_size=3,
                    padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_2d, self.hidden_dim_3d,
                kernel_size=(self.num_sample, 1, 1),
                stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3,
                    padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3,
                    padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

        self._get_interp1d_mask()
        self.bm_mask = self._get_bm_mask()
    
    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.Conv1d):
                kaiming_init(m)

    def _get_bm_mask(self):
        """Generate Boundary-Matching Mask."""
        bm_mask = []
        for idx in range(self.tscale):
            mask_vector = [1] * (self.tscale - idx) + [0] * idx
            bm_mask.append(mask_vector)
        bm_mask = torch.tensor(bm_mask, dtype=torch.float)
        return bm_mask

    def _get_interp1d_bin_mask(self,
                            seg_xmin,
                            seg_xmax,
                            tscale,
                            num_sample,
                            num_sample_perbin):
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[
                idx * num_sample_perbin: (idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if (int(sample_upper) <= (tscale - 1) and 
                                                    int(sample_upper) >= 0):
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        mask_mat = []
        for start_idx in range(self.tscale):
            mask_mat_vector = []
            for duration_idx in range(self.dscale):
                if start_idx + duration_idx < self.tscale:
                    p_min = start_idx
                    p_max = start_idx + duration_idx
                    center_len = float(p_max - p_min) + 1
                    sample_min = p_min - center_len * self.pbr
                    sample_max = p_max + center_len * self.pbr
                    p_mask = self._get_interp1d_bin_mask(
                        sample_min, sample_max, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(
            torch.Tensor(mask_mat).view(self.tscale, -1),
            requires_grad=False)

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(
            input_size[0], input_size[1],
            self.num_sample, self.tscale, self.tscale)
        return out

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        results = dict(confidence_map=confidence_map, start=start, end=end)
        return results

    def loss(self, inputs, label):
        pred_bm = inputs['confidence_map']
        pred_start = inputs['start']
        pred_end = inputs['end']
        device = pred_bm.device
        gt_iou_map = label['label_confidence'].to(device)
        gt_start = label['label_start'].to(device)
        gt_end = label['label_end'].to(device)
        losses = dict()

        pred_bm_reg = pred_bm[:, 0].contiguous()
        pred_bm_cls = pred_bm[:, 1].contiguous()
        gt_iou_map = gt_iou_map * self.bm_mask

        losses['tem_loss'] = self.tem_loss(
            pred_start, pred_end, gt_start, gt_end)
        losses['pem_reg_loss'] = self.pem_reg_loss(
            pred_bm_reg, gt_iou_map, self.bm_mask)
        losses['pem_cls_loss'] = self.pem_cls_loss(
            pred_bm_cls, gt_iou_map, self.bm_mask)
        return losses

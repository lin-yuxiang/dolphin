import math
import numpy as np
import torch
import torch.nn as nn

from dolphin.dataset.utils import ioa_with_anchors, iou_with_anchors
from dolphin.utils.postprocess import bmn_post_process
from dolphin.utils import Registers, base


@Registers.algorithm.register
class BoundaryMatchingNetwork(base.BaseAlgorithm):

    def __init__(self,
                 pretrained=None,
                 pretrained_modules=None,
                 backbone=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(BoundaryMatchingNetwork, self).__init__(
            backbone=backbone,
            pretrained_modules=pretrained_modules,
            head=head,
            pretrained=pretrained,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

        self.build_modules()

        self.tscale = self.head.temporal_scale
        self.temporal_gap = self.head.temporal_gap
        if test_cfg.get('soft_nms_alpha', None) is not None:
            self.soft_nms_alpha = test_cfg['soft_nms_alpha']
        if test_cfg.get('soft_nms_low_thres', None) is not None:
            self.soft_nms_low_thres = test_cfg['soft_nms_low_thres']
        if test_cfg.get('soft_nms_high_thres', None) is not None:
            self.soft_nms_high_thres = test_cfg['soft_nms_high_thres']
        if test_cfg.get('post_process_top_k', None) is not None:
            self.post_process_top_k = test_cfg['post_process_top_k']

        self.anchor_xmin = [
            self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [
            self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]
        
        self.init_weights()

    def forward_train(self, raw_feature, gt_bbox, **kwargs):
        label = self.generate_labels(gt_bbox)
        if self.backbone is not None:
            inputs = self.backbone(inputs)
        x = self.head(inputs)
        losses = self.neck.loss(x, label)
        return losses

    def forward_test(self, raw_feature, video_meta, **kwargs):
        assert hasattr(self, 'soft_nms_alpha')
        assert hasattr(self, 'soft_nms_low_thres')
        assert hasattr(self, 'soft_nms_high_thres')
        assert hasattr(self, 'post_process_top_k')
        if self.backbone is not None:
            raw_feature = self.backbone(raw_feature)
        inputs = self.head(raw_feature)
        confidence_map, start, end = inputs.values()
        start_scores = start[0].cpu().numpy()
        end_scores = end[0].cpu().numpy()
        cls_confidence = (confidence_map[0][1]).cpu().numpy()
        reg_confidence = (confidence_map[0][0]).cpu().numpy()

        new_props = []
        for idx in range(self.tscale):
            for jdx in range(self.tscale):
                start_index = idx
                end_index = jdx + 1
                if start_index < end_index and end_index < self.tscale:
                    xmin = start_index / self.tscale
                    xmax = end_index / self.tscale
                    xmin_score = start_scores[start_index]
                    xmax_score = end_scores[end_index]
                    cls_score = cls_confidence[idx, jdx]
                    reg_score = reg_confidence[idx, jdx]
                    score = xmin_score * xmax_score * cls_score * reg_score
                    new_props.append(
                        [xmin, xmax, xmin_score, xmax_score, cls_score,
                         reg_score, score])
        new_props = np.stack(new_props)
        video_info = dict(video_meta[0])
        proposal_list = bmn_post_process(
            new_props, video_info, self.soft_nms_alpha, self.soft_nms_low_thres,
            self.soft_nms_high_thres, self.post_process_top_k)
        output = [
            dict(video_name=video_info['video_name'],
            proposal_list=proposal_list)]
        return output

    def generate_labels(self, gt_bbox):
        label = dict()
        match_score_confidence_list = []
        match_score_start_list = []
        match_score_end_list = []
        for every_gt_bbox in gt_bbox:
            gt_xmins = every_gt_bbox[:, 0]
            gt_xmaxs = every_gt_bbox[:, 1]

            gt_iou_map = np.zeros([self.tscale, self.tscale])
            for i in range(self.tscale):
                for j in range(self.tscale):
                    gt_iou_map[i, j] = np.max(
                        iou_with_anchors(
                            i * self.temporal_gap, (j + 1) * self.temporal_gap,
                            gt_xmins, gt_xmaxs))
            gt_len_pad = 3 * self.temporal_gap
            gt_start_bboxs = np.stack(
                (gt_xmins - gt_len_pad / 2, gt_xmins + gt_len_pad / 2), axis=1)
            gt_end_bboxs = np.stack(
                (gt_xmaxs - gt_len_pad / 2, gt_xmaxs + gt_len_pad / 2), axis=1)
            
            match_score_start = []
            match_score_end = []

            for anchor_xmin, anchor_xmax in zip(
                self.anchor_xmin, self.anchor_xmax):
                match_score_start.append(
                    np.max(ioa_with_anchors(anchor_xmin, anchor_xmax, 
                           gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
                match_score_end.append(
                    np.max(ioa_with_anchors(anchor_xmin, anchor_xmax,
                           gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
            match_score_confidence_list.append(gt_iou_map)
            match_score_start_list.append(match_score_start)
            match_score_end_list.append(match_score_end)
        label['label_confidence'] = torch.Tensor(match_score_confidence_list)
        label['label_start'] = torch.Tensor(match_score_start_list)
        label['label_end'] = torch.Tensor(match_score_end_list)
        return label
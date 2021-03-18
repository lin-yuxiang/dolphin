from dolphin.base.base_model_module import BaseModelModule
from .assigner import MaxIoUAssigner
from .samplers import RandomSampler
from .roi_extractors import SingleRoIExtractor
from dolphin.utils import Registers, build_module_from_registers
from dolphin.utils.postprocess import bbox2result, bbox2roi


@Registers.head.register
class StandardRoIHead(BaseModelModule):

    def __init__(self,
                 bbox_roi_extractor=dict(
                     roi_layer=dict(
                         type='RoIAlign',
                         out_size=7,
                         sample_num=0),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 bbox_head=dict(
                     type='Shared2FCBBoxHead',
                     in_channels=256,
                     fc_out_channels=1024,
                     roi_feat_size=7,
                     num_classes=80,
                     bbox_coder=dict(
                         target_means=[0., 0., 0., 0.],
                         target_std=[0.1, 0.1, 0.1, 0.1]),
                     reg_class_agnostic=False,
                     loss_cls=dict(
                         type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=-1.0),
                     loss_bbox=dict(
                         type='L1Loss',
                         loss_weight=-1.0)),
                 shared_head=None,
                 assigner=dict(
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                 sampler=dict(
                    num=512,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposal=True),
                 pos_weight=-1,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.shared_head = shared_head
        self.pos_weight = pos_weight

        self.bbox_head = build_module_from_registers(
            bbox_head, module_name='head')

        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = SingleRoIExtractor(**bbox_roi_extractor)
        else:
            self.bbox_roi_extractor = None

        if assigner is not None:
            self.bbox_assigner = MaxIoUAssigner(**assigner)
        else:
            self.bbox_assigner = None
        if sampler is not None:
            self.bbox_sampler = RandomSampler(**sampler)
        else:
            self.bbox_sampler = None

    def init_weights(self, pretrained=None):
        if self.bbox_head is not None:
            self.bbox_head.init_weights(pretrained)
        if self.bbox_roi_extractor is not None:
            self.bbox_roi_extractor.init_weights(pretrained)

    def bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred)

    def forward(self,
                x,
                img_meta,
                proposal_list,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=None):
        num_imgs = len(img_meta)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self.bbox_forward(x, rois)
        bbox_results.update(dict(sampling_results=sampling_results, rois=rois))
        return bbox_results

    def loss(self, bbox_results, gt_bboxes, gt_labels):
        sampling_results = bbox_results['sampling_results']
        rois = bbox_results['rois']
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.pos_weight)
        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'], bbox_results['bbox_pred'], rois, 
            *bbox_targets)
        return loss_bbox

    def simple_test(self, x, proposal_list, img_meta, 
                    proposals_cfg=None, rescale=False):
        rois = bbox2roi(proposal_list)
        bbox_results = self.bbox_forward(x, rois)
        img_shape = img_meta[0]['shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=proposals_cfg)
        bbox_results = bbox2result(
            det_bboxes, det_labels, self.bbox_head.num_classes)
        return bbox_results
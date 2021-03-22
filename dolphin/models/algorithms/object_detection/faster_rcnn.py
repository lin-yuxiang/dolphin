from dolphin.utils import Registers, build_module_from_registers, base


@Registers.algorithm.register
class FasterRCNN(base.BaseAlgorithm):

    def __init__(self,
                 pretrained=None,
                 pretrained_modules=None,
                 backbone=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(FasterRCNN, self).__init__(
            pretrained=pretrained,
            pretrained_modules=pretrained_modules)

        self.backbone = build_module_from_registers(
            backbone, module_name='backbone')
        self.neck = build_module_from_registers(neck, module_name='neck')
        self.rpn_head = build_module_from_registers(
            rpn_head, module_name='head', 
            sub_cfg=dict(train_cfg=train_cfg, test_cfg=test_cfg))
        self.roi_head = build_module_from_registers(
            roi_head, module_name='head',
            sub_cfg=dict(train_cfg=train_cfg, test_cfg=test_cfg))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

    def forward_train(self, 
                      imgs, 
                      gt_bbox, 
                      label, 
                      img_meta, 
                      gt_bboxes_ignore=None,
                      proposals=None,
                      **kawrgs):
        x = self.backbone(imgs)
        if self.neck is not None:
            x = self.neck(x)
        losses = dict()
        if self.rpn_head is not None:
            x = self.rpn_head(x)
            proposal_cfg = self.train_cfg.get('rpn_proposal', None)
            rpn_losses, proposal_list = self.rpn_head.loss(
                x, img_meta, gt_bbox, label, gt_bboxes_ignore, proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        
        bbox_results = self.roi_head(
            x, img_meta, proposal_list, gt_bbox, label, 
            gt_bboxes_ignore=gt_bboxes_ignore)
        roi_losses = self.roi_head.loss(bbox_results, gt_bbox, label)
        losses.update(roi_losses)
        return losses

    def forward_test(self, 
                     imgs, 
                     img_meta, 
                     proposals=None, 
                     rescale=False, 
                     **kwargs):
        x = self.backbone(imgs)
        if self.neck is not None:
            x = self.neck(x)
        if proposals is None and self.rpn_head is not None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)
        else:
            proposal_list = proposals
        dets = self.roi_head.simple_test(
            x, proposal_list, img_meta, self.test_cfg['rcnn'], rescale=rescale)
        return dets
        
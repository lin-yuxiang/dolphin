from dolphin.utils import Registers
from dolphin.base.base_algorithm import BaseRecognizer


@Registers.algorithm.register
class Recognizer2D(BaseRecognizer):

    def __init__(self,
                 pretrained=None,
                 backbone=None,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(Recognizer2D, self).__init__()
        backbone_type = backbone.pop('type')
        self.backbone = Registers.backbone[backbone_type](**backbone)
        if neck is not None:
            neck_type = neck.pop('type')
            self.neck = Registers.neck[neck_type](**neck)
        else:
            self.neck = None
        cls_head_type = head.pop('type')
        self.cls_head = Registers.head[cls_head_type](**head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)
        if self.neck is not None:
            self.neck.init_weights(pretrained)
        self.cls_head.init_weights(pretrained)

    def extract_feat(self, imgs):
        x = self.backbone(imgs)
        return x

#     def forward_train(self, imgs, label, gt_bbox, img_meta):
#         # imgs = imgs.transpose(1, 0)
#         imgs = imgs.squeeze()
#         x = self.extract_feat(imgs)
#         x = self.neck(x)
#         cls_score = self.cls_head(x, gt_bbox, img_meta)
#         gt_labels = label.squeeze()
#         loss = self.cls_head.loss(cls_score.squeeze(), gt_labels)
#         return loss
    
    def forward_train(self, imgs, label, img_meta):
        # imgs = imgs.transpose(1, 0)
        imgs = imgs.squeeze()
        x = self.extract_feat(imgs)
        if self.neck is not None:
            x = self.neck(x)
        cls_score = self.cls_head(x, img_meta)
        gt_labels = label.squeeze()
        loss = self.cls_head.loss(cls_score.squeeze(), gt_labels)
        return loss

#     def forward_test(self, imgs, label, gt_bbox, img_meta):
#         # imgs = imgs.transpose(1, 0)
#         results = dict()
#         imgs = imgs.squeeze()
#         x = self.extract_feat(imgs)
#         x = self.neck(x)
#         cls_score = self.cls_head(x, gt_bbox, img_meta)
#         gt_labels = label.squeeze()
#         loss = self.cls_head.loss(cls_score.squeeze(), gt_labels)
#         results['val_loss'] = loss['loss_cls']
#         results['pred'] = cls_score
#         return results

    def forward_test(self, imgs, label, img_meta):
        # imgs = imgs.transpose(1, 0)
        results = dict()
        imgs = imgs.squeeze()
        x = self.extract_feat(imgs)
        if self.neck is not None:
            x = self.neck(x)
        cls_score = self.cls_head(x, img_meta)
        gt_labels = label.squeeze()
        loss = self.cls_head.loss(cls_score.squeeze(), gt_labels)
        results['val_loss'] = loss['loss_cls']
        results['pred'] = cls_score
        return results
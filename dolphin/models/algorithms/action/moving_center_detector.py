from dolphin.utils import Registers, base


@Registers.algorithm.register
class MovingCenterDetector(base.BaseAlgorithm):

    def __init__(self,
                 pretrained=None,
                 pretrained_modules=None,
                 backbone=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(MovingCenterDetector, self).__init__(
            pretrained=pretrained,
            pretrained_modules=pretrained_modules,
            backbone=backbone,
            head=head,
            train_cfg=None,
            test_cfg=None)

        self.build_modules()
        self.K = head['K']
        self.init_weights()

    def forward_train(self, imgs, label, **kwargs):
        x = self.extract_feat(imgs)
        if self.head is not None:
            x = self.head(x)
        losses = self.head.loss(x, label)
        return losses

    def forward_test(self, imgs, label, video_meta, **kwargs):
        x = self.extract_feat(imgs)
        if self.head is not None:
            x = self.head(x)
        detections = self.head.process(x, video_meta)
        return detections

    def extract_feat(self, imgs):
        chunk = [self.backbone(imgs[i]) for i in range(self.K)]
        return [chunk]
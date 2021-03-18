from dolphin.base.base_algorithm import BaseAlgorithm
from dolphin.utils import Registers


@Registers.algorithm.register
class ActivateLearning(BaseAlgorithm):

    def __init__(self,
                 pretrained=None,
                 pretrained_modules=None,
                 backbone=None,
                 head=None,
                 strategy=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(ActivateLearning, self).__init__(
            pretrained=pretrained,
            pretrained_modules=pretrained_modules,
            backbone=backbone,
            head=head,
            strategy=strategy,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        self.build_modules()
        self.init_weights()

    def forward_train(self, imgs, label, **kwargs):
        x = self.backbone(imgs)
        cls_score = self.head(x)
        loss = self.head.loss(cls_score, label)
        return loss

    def forward_test(self, imgs, **kwargs):
        x = self.backbone(imgs)
        cls_score = self.head(x)
        return cls_score

    def query(self, model, data_loader, mode, logger=None):
        data_loader.dataset.acquire_unlabeled_data()
        idx = self.strategy.query(model, data_loader, logger=logger)
        data_loader.dataset.update_label_set(idx)

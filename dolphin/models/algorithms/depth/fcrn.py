from dolphin.utils import Registers, base

@Registers.algorithm.register
class FCRN(base.BaseAlgorithm):

    def __init__(self,
                 pretrained=None,
                 pretrained_modules=None,
                 backbone=None,
                 decoder=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(FCRN, self).__init__(
            pretrained=pretrained,
            pretrained_modules=pretrained_modules,
            backbone=backbone,
            decoder=decoder,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        self.build_modules()
        self.init_weights()

    def forward_train(self, imgs, label, **kwargs):
        x = self.backbone(imgs)[0]
        x = self.decoder(x)
        x = self.head(x)
        loss = self.head.loss(x, label.unsqueeze(1))
        return loss

    def forward_test(self, imgs, label, **kwargs):
        x = self.backbone(imgs)[0]
        x = self.decoder(x)
        x = self.head(x)
        return x
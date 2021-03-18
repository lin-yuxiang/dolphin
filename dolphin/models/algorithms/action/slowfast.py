import torch.nn.functional as F

from dolphin.utils import Registers, base


@Registers.algorithm.register
class SlowFast(base.BaseAlgorithm):

    def __init__(self,
                 pretrained=None,
                 pretrained_modules=None,
                 backbone=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        
        super(SlowFast, self).__init__(
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            pretrained_modules=pretrained_modules,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

        self.build_modules()
        self.init_weights()

    def forward_train(self, imgs, label, **kwargs):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        gt_labels = label.squeeze()
        loss = self.cls_head.loss(cls_score, gt_labels)

        return loss

    def forward_test(self, imgs, label, **kwargs):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score)

        return cls_score.cpu().numpy()
    
    def average_clip(self, cls_score):

        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=1).mean(dim=0, keepdim=True)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=0, keepdim=True)
        return cls_score
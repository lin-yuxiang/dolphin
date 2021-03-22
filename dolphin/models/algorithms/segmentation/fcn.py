import torch.nn as nn
import torch.nn.functional as F

from dolphin.utils import Registers, build_module_from_registers, base


@Registers.algorithm.register
class FCN(base.BaseAlgorithm):

    def __init__(self,
                 pretrained=None,
                 pretrained_modules=None,
                 backbone=None,
                 head=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        
        super(FCN, self).__init__(
            pretrained=pretrained,
            pretrained_modules=pretrained_modules)

        self.backbone = build_module_from_registers(
            backbone, module_name='backbone')
        self.head = build_module_from_registers(head, module_name='head')
        self.auxiliary_head = build_module_from_registers(
            auxiliary_head, module_name='head')

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

    def forward_train(self, imgs, label, **kwargs):
        imgs = self.backbone(imgs)
        x = self.head(imgs)
        losses = self.head.loss(x, label)
        if self.auxiliary_head is not None:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.auxiliary_head):
                    x_aux = aux_head(imgs)
                    loss_aux = aux_head.loss(x_aux, label)
                    prefix = f'aux_{idx}'
                    new_loss_aux = dict()
                    for k, v in loss_aux:
                        new_loss_aux[prefix + k] = v
                    losses.update(new_loss_aux)
            else:
                x_aux = self.auxiliary_head(imgs)
                loss_aux = self.auxiliary_head.loss(x_aux, label)
                prefix = f'aux'
                new_loss_aux = dict()
                for k, v in loss_aux:
                    new_loss_aux[prefix + k] = v
                losses.update(new_loss_aux)
        return losses
        
    def forward_test(self, imgs, label, img_meta, **kwargs):
        assert self.test_cfg['mode'] in ['slide', 'whole']
        ori_shape = img_meta[0]['original_shape']
        assert all(_['original_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg['mode'] == 'slide':
            seg_logit = self.slide_inference(imgs, img_meta, rescale=True)
        else:
            seg_logit = self.whole_inference(imgs, img_meta, rescale=True)
        seg_logit = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_logit = seg_logit.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_logit = seg_logit.flip(dims=(2, ))
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg['stride']
        h_crop, w_crop = self.test_cfg['crop_size']
        batch_size, _, h_img, w_img = img.size()
        assert h_crop <= h_img and w_crop <= w_img, (
            'crop size should not greater than image size')
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.backbone(crop_img)
                crop_seg_logit = self.head(crop_seg_logit)
                crop_seg_logit = F.interpolate(
                    crop_seg_logit,
                    size=crop_img.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                preds += F.pad(
                    crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1),
                    int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = F.interpolate(
                preds,
                size=img_meta[0]['original_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.backbone(img)
        seg_logit = self.head(seg_logit)
        seg_logit = F.interpolate(
            seg_logit,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        if rescale:
            seg_logit = F.interpolate(
                seg_logit,
                size=img_meta[0]['original_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit
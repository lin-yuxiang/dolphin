import os.path as osp
import numpy as np
from functools import reduce

from ..base import BaseDataset
from dolphin.utils import Registers, scandir
from dolphin.dataset.utils import imread
from dolphin.evaluate import mean_iou


@Registers.data.register
class PascalVOCDataset(BaseDataset):

    CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    def __init__(self,
                 ann_file=None,
                 pipeline=None,
                 img_dir='JPEGImages',
                 ann_dir='SegmentationClass',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 reduce_zero_label=False,
                 data_prefix=None,
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(PascalVOCDataset, self).__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.reduce_zero_label = reduce_zero_label

        if self.data_prefix is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_prefix, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_prefix, self.ann_dir)
            if not (self.ann_file is None or osp.isabs(self.ann_file)):
                self.ann_file = osp.join(self.data_prefix, self.ann_file)

    def load_annotations(self):
        img_infos = []
        if self.ann_file is not None:
            with open(self.ann_file) as f:
                for line in f:
                    img_name = line.strip()
                    img_file = osp.join(
                        self.img_dir, img_name + self.img_suffix)
                    img_info = dict(filename=img_file)
                    if self.ann_dir is not None:
                        seg_map = osp.join(
                            self.ann_dir, img_name + self.seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in scandir(self.img_dir, self.img_suffix, recursive=True):
                img_file = osp.join(self.img_dir, img)
                img_info = dict(filename=img_file)
                if self.ann_dir is not None:
                    seg_map = osp.join(
                        self.ann_dir, img.replace(
                            self.img_suffix, self.seg_map_suffix))
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        return img_infos
    
    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.data_infos:
            gt_seg_map = imread(img_info['ann']['seg_map'], flag='unchanged')
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        all_acc, acc, iou = mean_iou(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Acc')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for i in range(num_classes):
            iou_str = '{:.2f}'.format(iou[i] * 100)
            acc_str = '{:.2f}'.format(acc[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, acc_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')

        iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
        acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
        all_acc_str = '{:.2f}'.format(all_acc * 100)
        summary_str += line_format.format(
            'global', iou_str, acc_str, all_acc_str)
        logger.info(summary_str)

        eval_results['mIoU'] = np.nanmean(iou)
        eval_results['mAcc'] = np.nanmean(acc)
        eval_results['aAcc'] = all_acc

        return eval_results

    def prepare_train_data(self, idx):
        img_info = self.data_infos[idx]
        filename = img_info['filename']
        ann_info = self.data_infos[idx]['ann']
        results = dict(filename=filename, ann_info=ann_info)
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        img_info = self.data_infos[idx]
        filename = img_info['filename']
        results = dict(filename=filename)
        return self.pipeline(results)
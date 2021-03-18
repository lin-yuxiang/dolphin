import torch
import copy
import os.path as osp

from dolphin.dataset.evaluate import mean_average_precision, mean_class_accuracy
from dolphin.loss import top_k_accuracy
from ..base import BaseDataset
from dolphin.utils import Registers


@Registers.data.register
class RawframeDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 multi_class=False,
                 num_classes=None,
                 modality='RGB',
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(RawframeDataset, self).__init__(
            ann_file, 
            pipeline, 
            data_prefix, 
            test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

        self.filename_tmpl = filename_tmpl
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.modality = modality

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.multi_class is not None
                    (frame_dir, total_frames,
                    label) = (line_split[0], line_split[1], line_split[2:])
                    label = list(map(int, label))
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                else:
                    frame_dir, total_frames, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_infos.append(dict(
                        frame_dir=frame_dir,
                        total_frames=int(total_frames),
                        label=onehot if self.multi_class else label))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if not isinstance(topk, (int, tuple)):
            raise TypeError(
                f'topk must be int or tuple of int, but got {type(topk)}')

        if isinstance(topk, int):
            topk = (topk, )

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision'
        ]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            logger.info(f'\nEvaluating {metric}...')

            if metric == 'top_k_accuracy':
                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                logger.info(log_msg)
                continue

            if metric == 'mean_average_precision':
                gt_labels = [label.cpu().numpy() for label in gt_labels]
                mAP = mean_average_precision(results, gt_labels)
                eval_results['mean_average_precision'] = mAP
                log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                logger.info(log_msg)
                continue

        return eval_results
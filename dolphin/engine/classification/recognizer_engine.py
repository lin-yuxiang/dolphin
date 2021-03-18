import time
import torch
import numpy as np
import os.path as osp
from sklearn.metrics import roc_curve, auc

from ..base import ABCEngine
from dolphin.utils import Registers, Bar


@Registers.engine.register
class RecognizerEngine(ABCEngine):

    def __init__(self,
                 algorithm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 data=None,
                 runtime_cfg=None,
                 meta=None,
                 distributed=False):

        super(RecognizerEngine, self).__init__(
            algorithm_cfg=algorithm_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data=data,
            runtime_cfg=runtime_cfg,
            meta=meta,
            distributed=distributed)
    
    def val(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode)
        val_batches = len(data_loader)

        bar = Bar(
            'Val Epoch {}'.format(self.epoch), 
            max=(val_batches // self.vis_interval))
        self.meters.before_epoch()

        preds = []
        labels = []
        for i, data_batch in enumerate(data_loader):
            start_time = time.time()
            self._inner_iter = i
            with torch.no_grad():
                outputs = self.model.test_step(
                    data_batch, self.optimizer, **kwargs)
                results = outputs['results']
                pred = results['pred']
                preds.append(pred.detach().cpu().squeeze().numpy())
                labels.append(
                    data_batch['label'].detach().cpu().squeeze().numpy()) 
            end_time = time.time()
            self.meters.update({'loss': results['val_loss'].item()})
            self.meters.update({'batch_time': end_time - start_time})
            self.meters.during_train_iter(self._inner_iter, self.vis_interval)

            # self._iter += 1
            if i % self.vis_interval == 0:
                print_str = '[{}/{}]'.format(
                    (i // self.vis_interval) + 1, 
                    val_batches // self.vis_interval)
                for name, val in self.meters.output.items():
                    print_str += ' | {} {:.6f}'.format(name, val)
                Bar.suffix = print_str
                bar.next()
        bar.finish()
        self.summary[f'epoch_{self._epoch}']['val_loss'] = \
            self.meters.output['loss']
        self.meters.after_val_epoch()

        preds = np.array(preds)
        labels = np.array(labels)
        fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
        auc_val = auc(fpr, tpr)
        
        thres = 0.5
        neg_labels = (labels == 0).astype(np.int)
        pred_labels = (preds >= thres).astype(np.int)
        tp = sum(labels * (pred_labels == 1))
        fp = sum(neg_labels * (pred_labels == 1))
        fn = sum(labels * (pred_labels == 0))
        # precision:
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        self.log.info(
            f'AUC Value: {auc_val} | Precision: {precision} | Recall: {recall}')
        self.summary[f'epoch_{self._epoch}']['auc'] = auc_val
        self.summary[f'epoch_{self._epoch}']['precision'] = precision
        self.summary[f'epoch_{self._epoch}']['recall'] = recall
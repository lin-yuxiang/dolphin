import torch
import time
import os.path as osp

from ..base import ABCEngine
from dolphin.utils import Registers, Bar
from dolphin import utils


@Registers.engine.register
class ActionDetectionEngine(ABCEngine):

    def __init__(self,
                 algorithm=None,
                 train_cfg=None,
                 test_cfg=None,
                 data=None,
                 runtime_cfg=None,
                 meta=None,
                 distributed=False):

        super(ActionDetectionEngine, self).__init__(
            algorithm=algorithm,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data=data,
            runtime_cfg=runtime_cfg,
            meta=meta,
            distributed=distributed)

    def test(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode)
        num_batch = len(data_loader)
        bar = Bar('Extracting results ', max=(num_batch // self.vis_interval))
        self.meters.before_epoch()

        for i, data_batch in enumerate(data_loader):
            start_time = time.time()
            self._inner_iter = i
            with torch.no_grad():
                outputs = self.model.test_step(data_batch, **kwargs)
                results = outputs['results']
                num_samples = outputs['num_samples']
            end_time = time.time()

            inference_dir = self.test_cfg['inference_dir']
            if inference_dir is not None:
                data_loader.dataset.dump_results(
                    results, data_batch, num_samples)
            else:
                raise ValueError(
                    'inference_dir should not be None when using dump_results.')

            self.meters.update({'batch_time': end_time - start_time})
            self.meters.after_val_iter(self._inner_iter, self.vis_interval)

            if i % self.vis_interval == 0:
                print_str = '[{}/{}]'.format(
                    (i // self.vis_interval) + 1, 
                    num_batch // self.vis_interval)
                for name, val in self.meters.output.items():
                    print_str += ' | {} {:.4f}'.format(name, val)
                Bar.suffix = print_str
                bar.next()
        bar.finish()
        self.meters.after_val_epoch()
        data_loader.dataset.evaluate(
            metric=self.test_cfg['metric'], logger=self.log)
        self._epoch += 1
import torch
import time
import os.path as osp

from ..base import ABCEngine
from dolphin.utils import Registers, Bar
from dolphin import utils


@Registers.algorithm.register
class ActionRecognizerEngine(object):

    def __init__(self,
                 algorithm=None,
                 train_cfg=None,
                 test_cfg=None,
                 data=None,
                 runtime_cfg=None,
                 meta=None,
                 distributed=False):

        super(ActionRecognizerEngine, self).__init__(
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
        results = []

        for i, data_batch in enumerate(data_loader):
            start_time = time.time()
            # during train iter
            with torch.no_grad():
                outputs = self.model.test_step(data_batch, **kwargs)
            # after train iter
            outputs = outputs['results']
            results.extend(outputs)

            end_time = time.time()
            self.meters.update({'batch_time': end_time - start_time})
            self.meters.average()

            if i % self.vis_interval == 0:
                print_str = '[{}/{}]'.format(
                    (i // self.vis_interval) + 1, 
                    num_batch // self.vis_interval)
                for name, val in self.meters.output.items():
                    print_str += ' | {} {:.4f}'.format(name, val)
                Bar.suffix = print_str
                self.log.debug(f'Extracting results :' + print_str)
                bar.next()
            self.meters.clear_output()        
        bar.finish()
        
        eval_results = data_loader.dataset.evaluate(
            results, metrics=self.test_cfg['metrics'], logger=self.log)
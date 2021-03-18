import torch
import time
import numpy as np
import os.path as osp

from ..base import ABCEngine
from dolphin.utils import Registers, Bar
from dolphin import utils
from dolphin.dataset import PCLDataParallel


@Registers.engine.register
class ActivateLearningEngine(ABCEngine):

    def __init__(self,
                 algorithm=None,
                 train_cfg=None,
                 test_cfg=None,
                 data=None,
                 runtime_cfg=None,
                 meta=None,
                 distributed=False,
                 **kwargs):

        super(ActivateLearningEngine, self).__init__(
            algorithm=algorithm,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data=data,
            runtime_cfg=runtime_cfg,
            meta=meta,
            distributed=distributed)

    def query(self, data_loader, mode, **kwargs):
        self.log.info(f'Start query mode...')
        self.model.set_mode(mode)
        if hasattr(self.model, 'module'):
            func = self.model.module.query
        else:
            func = self.model.query
        func(self.model, data_loader, mode, self.log)
        self.log.info(f'Labeled dataset updated sucessfully.')

    def test(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode)

        num_batch = len(data_loader)
        bar = Bar('Extracting results ', max=(num_batch // self.vis_interval))
        self.meters.before_epoch()
        pred = []

        for i, data_batch in enumerate(data_loader):
            start_time = time.time()
            # during train iter
            with torch.no_grad():
                outputs = self.model.test_step(data_batch, **kwargs)
            # after train iter
            results = outputs['results']
            pred.append(results.max(1)[1].cpu())

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
        
        pred = np.array(pred)
        data_loader.dataset.evaluate(pred, self.log)
import torch
import time
import os.path as osp

from ..base import ABCEngine
from dolphin.utils import Registers, Bar
from dolphin import utils
from dolphin.models.utils import (load_checkpoint, load_npy_weights, 
                                load_state_dict)


@Registers.engine.register
class DepthEngine(ABCEngine):

    def __init__(self,
                 algorithm=None,
                 train_cfg=None,
                 test_cfg=None,
                 data=None,
                 runtime_cfg=None,
                 meta=None,
                 distributed=False):

        super(DepthEngine, self).__init__(
            algorithm=algorithm,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data=data,
            runtime_cfg=runtime_cfg,
            meta=meta,
            distributed=distributed)

    def test(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode)
        test_batches = len(data_loader)

        thresh_1_25 = 0
        thresh_1_25_2 = 0
        thresh_1_25_3 = 0
        rmse_linear = 0.0
        rmse_log = 0.0
        rmse_log_scale_invariant = 0.0
        ard = 0.0
        srd = 0.0

        bar = Bar('Testing ', max=(test_batches // self.vis_interval))
        self.meters.before_epoch()
        self.model.eval()
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.test_step(data_batch, **kwargs)
                results = outputs['results']
                # num_samples = outputs['num_samples']
            end_time = time.time()

            self.meters.update({'batch_time': end_time - start_time})
            self.meters.after_val_iter(self._inner_iter, self.vis_interval)

            out = data_loader.dataset.evaluate(
                self._inner_iter + 1, 
                results, 
                data_batch,
                test_batches,
                thresh_1_25,
                thresh_1_25_2,
                thresh_1_25_3,
                rmse_linear,
                rmse_log,
                rmse_log_scale_invariant,
                ard,
                srd)
            thresh_1_25 += out[0]
            thresh_1_25_2 += out[1]
            thresh_1_25_3 += out[2]
            rmse_linear += out[3]
            rmse_log += out[4]
            rmse_log_scale_invariant += out[5]
            ard += out[6]
            srd += out[7]

            if i % self.vis_interval == 0:
                print_str = '[{}/{}]'.format((i // self.vis_interval) + 1,
                                    test_batches // self.vis_interval)
                for name, val in self.meters.output.items():
                    print_str += ' | {} {:.4f}'.format(name, val)
                Bar.suffix = print_str
                bar.next()
        bar.finish()
        self.meters.after_val_epoch()
        
        self.log.info('\nThreshold_1.25: {}'.format(thresh_1_25))
        self.log.info('\nThreshold_1.25^2: {}'.format(thresh_1_25_2))
        self.log.info('\nThreshold_1.25^3: {}'.format(thresh_1_25_3))
        self.log.info('\nRMSE_linear: {}'.format(rmse_linear))
        self.log.info('\nRMSE_log: {}'.format(rmse_log))
        self.log.info('\nRMSE_log_scale_invariant: {}'.format(
            rmse_log_scale_invariant))
        self.log.info('\nARD: {}'.format(ard))
        self.log.info('\nSRD: {}'.format(srd))

        self._epoch += 1

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.log.info('load checkpoint from %s', filename)
        if filename.endswith('npy'):
            dtype = torch.cuda.FloatTensor
            if hasattr(self.model, 'module'):
                state_dict = load_npy_weights(
                    self.model.module, filename, dtype)
                load_state_dict(self.model.module, state_dict, strict)
            else:
                state_dict = load_npy_weights(self.model, filename, dtype)
                load_state_dict(self.model, state_dict, strict)
        else:
            return load_checkpoint(self.model, filename, map_location, strict)
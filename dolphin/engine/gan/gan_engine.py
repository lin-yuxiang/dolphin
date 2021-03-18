import torch
import os.path as osp
import os
import itertools
import time

from ..base import ABCEngine
from dolphin.utils import Registers, Bar 
from dolphin import utils
from dolphin.dataset import PCLDataParallel
from dolphin.models.utils import load_checkpoint, save_checkpoint
from dolphin.engine.solver import lr_scheduler


@Registers.engine.register
class CycleGANEngine(ABCEngine):

    def __init__(self,
                 algorithm=None,
                 train_cfg=None,
                 test_cfg=None,
                 data=None,
                 runtime_cfg=None,
                 meta=None,
                 distributed=None):

        super(CycleGANEngine, self).__init__(
            algorithm=algorithm,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data=data,
            runtime_cfg=runtime_cfg,
            meta=meta,
            distributed=distributed)

    def build_optimizer(self, model, runtime_cfg):
        assert isinstance(runtime_cfg['optimizer'], dict)
        optim_G_cfg = runtime_cfg['optimizer']['G']
        optim_D_cfg = runtime_cfg['optimizer']['D']
        optim_G_type = optim_G_cfg.pop('type')
        optim_D_type = optim_D_cfg.pop('type')
        if hasattr(model, 'module'):
            model = model.module
        assert hasattr(model, 'netG_A') and hasattr(model, 'netG_B')
        optim_G_cfg['params'] = itertools.chain(
            model.netG_A.parameters(), model.netG_B.parameters())
        assert hasattr(model, 'netD_A') and hasattr(model, 'netD_B')
        optim_D_cfg['params'] = itertools.chain(
            model.netD_A.parameters(), model.netD_B.parameters())
        optimizer_G = getattr(torch.optim, optim_G_type)(**optim_G_cfg)
        optimizer_D = getattr(torch.optim, optim_D_type)(**optim_D_cfg)
        return dict(G=optimizer_G, D=optimizer_D)

    def build_lr_scheduler(self, runtime_cfg):
        assert isinstance(runtime_cfg['lr_config'], dict)
        scheduler_G_cfg = runtime_cfg['lr_config']['G']
        scheduler_D_cfg = runtime_cfg['lr_config']['D']
        scheduler_G_type = lr_scheduler.get_scheduler_type(scheduler_G_cfg)
        scheduler_D_type = lr_scheduler.get_scheduler_type(scheduler_D_cfg)
        lr_scheduler_G = getattr(
            lr_scheduler, scheduler_G_type)(**scheduler_G_cfg)
        lr_scheduler_D = getattr(
            lr_scheduler, scheduler_D_type)(**scheduler_D_cfg)
        return dict(G=lr_scheduler_G, D=lr_scheduler_D)
    
    def train(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode)
        self._epoch += 1
        num_batches = len(data_loader)
        # self._max_iter = self._max_epoch * train_batches

        self.adjust_lr_by(num_batches, self.epoch, self.cur_iter, phase='epoch')

        bar = Bar(
            'Train Epoch {}'.format(self.epoch), 
            max=(num_batches // self.vis_interval))
        self.meters.before_epoch()

        for i, data_batch in enumerate(data_loader):
            start_time = time.time()
            # during train iter
            self._inner_iter = i
            self.adjust_lr_by(
                num_batches, self.epoch, self.cur_iter, phase='iter')

            outputs = self.model.train_step(
                data_batch, optimizer=self.optimizer, **kwargs)
            self.meters.update(outputs['log_vars'], outputs['num_samples'])
            # after train iter

            end_time = time.time()
            self.meters.update({'batch_time': end_time - start_time})
            self.meters.during_train_iter(self._inner_iter, self.vis_interval)

            self._iter += 1
            if i % self.vis_interval == 0:
                print_str = '[{}/{}]'.format(
                    (i // self.vis_interval) + 1, 
                    num_batches // self.vis_interval)
                for name, val in self.meters.output.items():
                    print_str += ' | {} {:.6f}'.format(name, val)
                Bar.suffix = print_str
                self.log.debug(f'Epoch: {self.epoch}' + print_str)
                bar.next()
            # self.meters.after_train_iter()
        
        bar.finish()
        # TODO: SUMMARY NOT COMPLETE
        self.summary[f'epoch_{self._epoch}']['train_loss'] = \
            self.meters.output['loss']
        self.meters.after_train_epoch()
        self.checkpointer.after_train_epoch(self)
    
    def test(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode)
        
        num_batch = len(data_loader)
        bar = Bar('Extracting results ', max=(num_batch // self.vis_interval))
        self.meters.before_epoch()

        for i, data_batch in enumerate(data_loader):
            start_time = time.time()
            # during train iter
            with torch.no_grad():
                outputs = self.model.test_step(data_batch, **kwargs)
            # after train iter
            outputs = outputs['results']
            filename = data_batch['filename']
            direction = self.algorithm['direction']
            img_path = [filename['A'] if direction == 'AtoB' else filename['B']]

            end_time = time.time()
            self.meters.update({'batch_time': end_time - start_time})
            self.meters.average()

            save_dir = osp.join(self.work_dir, 'results')
            utils.mkdir_or_exist(save_dir)
            short_path = osp.basename(img_path[0])
            name = osp.splitext(short_path)[0]
            for label, output in outputs.items():
                output = utils.tensor2img(output)
                image_name = '%s_%s.png' % (name, label)
                save_path = osp.join(save_dir, image_name)
                utils.imwrite(output, save_path)

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
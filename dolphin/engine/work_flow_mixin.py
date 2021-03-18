import time
import json
import torch
import os.path as osp

from dolphin.utils import Bar

class WorkFlowMixIn(object):

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
            self.optimizer.zero_grad()
            outputs['loss'].backward()
            self.optimizer.step()

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

    def val(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode)
        val_batches = len(data_loader)

        bar = Bar(
            'Val Epoch {}'.format(self.epoch), 
            max=(val_batches // self.vis_interval))
        self.meters.before_epoch()

        for i, data_batch in enumerate(data_loader):
            start_time = time.time()
            self._inner_iter = i
            with torch.no_grad():
                outputs = self.model.val_step(data_batch, **kwargs)
            outputs = outputs['results']
            self.meters.update(outputs['log_vars'], outputs['num_samples'])
            end_time = time.time()
            self.meters.update({'batch_time': end_time - start_time})
            self.meters.during_train_iter(self._inner_iter, self.vis_interval)

            # self._iter += 1
            if i % self.vis_interval == 0:
                print_str = '[{}/{}]'.format(
                    (i // self.vis_interval) + 1, 
                    val_batches // self.vis_interval)
                for name, val in self.meters.output.items():
                    print_str += ' | {} {:.4f}'.format(name, val)
                Bar.suffix = print_str
                bar.next()
        bar.finish()
        self.meters.after_val_epoch()
        # TODO: SUMMARY NOT COMPLETE
        self.summary[f'epoch_{self._epoch}']['val_loss'] = \
            self.meters.output['loss'].detach().cpu().data

    def test(self):
        pass

    def run(self, **kwargs):
        assert isinstance(self.data_loaders, dict)
        assert isinstance(self.workflow, list)

        test_mode = False
        self._max_epoch = self.total_epochs
        self.init_summary()

        for flow in self.workflow:
            mode, duration = flow
            if mode == 'train':
                self._max_iter = self._max_epoch * len(self.data_loaders[mode])
                break
            elif mode == 'test':
                assert duration == 1
                self.total_epochs = duration
                test_mode = True

        work_dir = self.work_dir if self.work_dir is not None else 'None'
        self.log.info(f'Start running, work_dir: {work_dir}')
        self.log.info(f'workflow: {self.workflow}, max: {self.total_epochs}')

        if not test_mode:
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, dict):
                    for key, value in self.lr_scheduler.items():
                        value.initialize_lr(self.optimizer[key])
                else:
                    self.lr_scheduler.initialize_lr(self.optimizer)
            self.meters.before_run()

        while self.epoch < self.total_epochs:
            for _, flow in enumerate(self.workflow):
                mode, epochs = flow
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                    f'mode in workflow must be str, but got {type(mode)}')
                
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self.total_epochs:
                        return
                    if mode == 'query':
                        data_loader = self.data_loaders['train']
                    else:
                        data_loader = self.data_loaders[mode]
                    epoch_runner(data_loader, mode, **kwargs)

        # TODO: flag设置在config文件中
        with open(osp.join(self.work_dir, 'summary.json'), 'a') as f:
            json.dump(self.summary, f)
        self.log.info('Summary file written.')
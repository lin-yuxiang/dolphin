def get_scheduler_type(lr_config):
    assert isinstance(lr_config, dict) and 'policy' in lr_config
    policy_type = lr_config.pop('policy')
    if policy_type == policy_type.lower():
        policy_type = policy_type.title()

    scheduler_type = policy_type + 'LrScheduler'
    return scheduler_type

class LrScheduler(object):

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                'warmup_iters must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                'warmup_ratio must be in range (0, 1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []
        self.regular_lr = []

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def get_lr(self, epoch, iters, base_lr):
        raise NotImplementedError

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def get_regular_lr(self, optimizer, epoch, iters):
        if isinstance(optimizer, dict):
            lr_groups = {}
            for k in optimizer.keys():
                _lr_group = [
                    self.get_lr(epoch, iters, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})
            return lr_groups
        else:
            return [self.get_lr(epoch, iters, _base_lr)
                        for _base_lr in self.base_lr]

    def initialize_lr(self, optimizer):
        if isinstance(optimizer, dict):
            self.base_lr = {}
            for k, optim in optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr'] for group in optimizer.param_groups
            ]

    def adjust_by_epoch(self, batch_num, optimizer, epoch, iters):
        if not self.by_epoch:
            return
        if self.warmup_by_epoch:
            self.warmup_iters = self.warmup_epochs * batch_num
        
        self.regular_lr = self.get_regular_lr(optimizer, epoch, iters)
        self._set_lr(optimizer, self.regular_lr)

    def adjust_by_iter(self, epoch, cur_iter, optimizer):
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(optimizer, epoch, cur_iter)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(optimizer, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(optimizer, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(optimizer, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(optimizer, warmup_lr)


class StepLrScheduler(LrScheduler):

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('step must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrScheduler, self).__init__(**kwargs)
    
    def get_lr(self, epoch, iters, base_lr):
        progress = epoch if self.by_epoch else iters

        if isinstance(self.step, int):
            return base_lr * (self.gamma ** (progress // self.step))
        
        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma ** exp
from collections import OrderedDict
import numpy as np


class Meters(object):
    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False
        self.reset_flag = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()
    
    def clear_output(self):
        self.output.clear()
        self.ready = False
    
    def update(self, vals, count=1):
        assert isinstance(vals, dict)
        for key, val in vals.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(val)
            self.n_history[key].append(count)

    def average(self, n=0):
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])    
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True

    def before_run(self):
        self.reset_flag = True

    def before_epoch(self):
        self.clear()

    def during_train_iter(self, inner_iter, interval):
        if (inner_iter + 1) % interval == 0:
            self.average(interval)

    def after_train_iter(self):
        if self.ready:
            if self.reset_flag:
                self.clear_output()
    
    def after_val_iter(self, inner_iter, interval):
        if (inner_iter + 1) % interval == 0:
            self.average(interval)
        
        if self.ready:
            if self.reset_flag:
                self.clear_output()

    def after_train_epoch(self):
        if self.ready:
            if self.reset_flag:
                self.clear_output()
    
    def after_val_epoch(self):
        self.average()
        if self.reset_flag:
            self.clear_output()
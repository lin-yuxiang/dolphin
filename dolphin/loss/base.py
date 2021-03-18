from abc import ABCMeta, abstractmethod

import torch.nn as nn

class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):

    def __init__(self, loss_weight=1.0, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        self.kwargs.update(kwargs)
        return self._forward(*args, **self.kwargs) * self.loss_weight
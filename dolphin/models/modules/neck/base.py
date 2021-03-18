import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class BaseNeck(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super(BaseNeck, self).__init__()

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self):
        pass
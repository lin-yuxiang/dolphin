import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from .base_module import BaseModule


class BaseModelModule(BaseModule, nn.Module, metaclass=ABCMeta):

    def __init__(self):
        BaseModule.__init__(self)
        nn.Module.__init__(self)
    
    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, nn.Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            dolphin_modules = self.__dict__.get('_dolphin_modules')
            if isinstance(value, nn.Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
                if isinstance(value, BaseModule):
                    if dolphin_modules is None:
                        raise AttributeError(
                        "cannot assign module before BaseModule.__init__().")
                    dolphin_modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
                if dolphin_modules is not None and name in dolphin_modules:
                    if value is not None:
                        raise TypeError("cannot assign '{}' as child module"
                                        " '{}' (BaseModule or None expected)"
                                        .format(torch.typename(value), name))
                    dolphin_modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(
                        value, torch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)
        
    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
            if name in self._dolphin_modules:
                del self._dolphin_modules[name]
        else:
            object.__delattr__(self, name)
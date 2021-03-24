from collections import OrderedDict


class BaseModule():
    '''
    Base Module of Open Dolphin, which inherited from all modules.
    '''
    
    def __init__(self):
        self._dolphin_modules = OrderedDict()

    def __getattr__(self, name):
        if '_dolphin_modules' in self.__dict__:
            _dolphin_modules = self.__dict__['_dolphin_modules']
            if name in _dolphin_modules:
                return _dolphin_modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        _dolphin_modules = self.__dict__.get('_dolphin_modules')
        if isinstance(value, BaseModule):
            if _dolphin_modules is None:
                raise AttributeError(
                    "cannot assign modules before BaseModule.__init__().")
            _dolphin_modules[name] = value
        elif _dolphin_modules is not None and name in _dolphin_modules:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as child module"
                    "(BaseModule or None expected)".format(name))
            _dolphin_modules[name] = value
        else:
            object.__setattr__(self, name, value)
    
    def __delattr__(self, name):
        if name in self._dolphin_modules:
            del self._dolphin_modules[name]
        else:
            object.__delattr__(self, name)

    def dolphin_modules(self):
        for name, module in self.named_dolphin_modules():
            yield name, module

    def named_dolphin_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._dolphin_modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_dolphin_modules(memo, submodule_prefix):
                    yield m
                
    def named_pcl_children(self):
        memo = set()
        for name, module in self._dolphin_modules.items():
            if module is not None and module not in memo:
                yield name, module
    
    def pcl_children(self):
        for name, module in self.named_pcl_children():
            yield module

    def set_mode(self, mode=None):
        if mode is None:
            raise ValueError("parameter 'mode' should not be NoneType.")
        self.mode = mode
        for children in self.pcl_children():
            children.set_mode(mode)
        if hasattr(self, 'train') and hasattr(self, 'training'):
            if 'train' in mode:
                self.train()
            else:
                self.eval()
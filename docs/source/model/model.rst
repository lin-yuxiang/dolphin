
.. _submodule_model:

Submodule of Model
==================

This module contains some submodules from model, for example backbones, heads and
so on. A new submodule of model must inherit from **BaseModelAlgorithm** which is
from **BaseModule**. Due to the base framework of this project is PyTorch, so the
model components need to inherit from *torch.nn.Module*. The API can be checked below:

.. code-block:: python

    import torch.nn as nn
    from abc import ABCMeta, abstractmethod

    from dolphin.utils import base

    class BaseModelModule(base.BaseModule, nn.Module, metaclass=ABCMeta):
        
        def __init__(self):
            BaseModule.__init__()
            nn.Module.__init__()
        
        @abstractmethod
        def forward(self):
            pass

        @abstractmethod
        def init_weights(self):
            pass

.. tip::
    Here only present the API of BaseModelModule, the example of customization 
    of submodule is showed in :ref:`Management <management>`.


.. _algorithm:

Algorithm
=========

Module *Algorithm* plays an role on combinations of all components from a model. 
A new algorithm module must inherit from **BaseAlgorithm** which is from **BaseModelModule**, 
whose member funtions *forward_train* and *forward_test* should be implemented:

.. code-block:: python

    from dolphin.utils import base

    class BaseAlgorithm(base.BaesModelModule):

        def __init__(self, *args, **kwargs):
            super(BaseAlgorithm, self).__init__()

        @abstractmethod
        def forward_train(self):
            pass

        @abstractmethod
        def forward_test(self):
            pass

        def build_modules(self):
            '''
            Member function of class BaseAlgorithm which is capable
            to build submodules of algorithm automatically according to
            arguments provided during creation of instance.
            '''

        def init_weights(self):
            '''
            Recursively initialize weights of every submodule.
            Methods with the same name of submodules needed to be implemented.
            '''


.. _utils_model:

Utility
=======

Some commonly and widely use funtions are keep here. For example the loading of
pretrained checkpoint: *load_checkpoint*, saving of checkpoint **save_checkpoint**
and so on.
.. _mixin:

Mix-In Methods
==============

The file named *work_flow_mixin.py* presented as a class, which contains some 
commonly methods of training, testing and so on. For easily management methods of
engines, they are departed independent.


.. _solver:

Solver
======

Here stores helpful components during training phase or testing phase, such as
optimizer, learning rate scheduler and so on. New optimizer and scheduler can be
put in here. Different from modules, implementation of these components isn't
necessary to register as a module. They can be called by *engine* module once their
names are assigned in *config file*.

.. _base_engine:

Base Engine
===========

Module *Engine* inherits from **BaseModule** and **WorkFlowMixIn** (work_flow_mixin.py),
which is the kernel of all modules. Normally for different algorithms, only the method 
of training and testing needed to be newly implemented, so for convenient these kind of
methods were moved into class **WorkFlowMixIn**. Its API is shown below:

.. note::
    For customization of engine class, it is also necessary to register it to **Registry**
    for use.

.. code-block:: python

    class Engine(BaseModule, WorkFlowMixIn, metaclass=ABCMeta):
    
        '''
        Module takes in charge of organize modules and process.

        Inputs: algorithm (dict): algorithm configuration.
                data (dict): dataset configuration.
                cfg (dict): other configurations. Such as phases, optimizer name
                            and so on.
        '''

        def __init__(self,
                     algorithm=None,
                     data=None,
                     cfg=None,
                     **kwargs):

            self.init_engine()
            self.set_logger()
        
        def init_engine(self):
            '''
            initialize the engine with its input argument.
            '''

        def set_logger(self):
            '''
            set up logger.
            '''
        
        def build_models(self):
            '''
            building algorithm, schedule them as a property of engine.
            '''

        def build_dataset(self):
            '''
            building dataset, schedule it as a property of engine.
            '''

        def build_optimizer(self):
            '''
            build optimizer when training.
            '''

        def resume(self, filename, resume_optimizer=True):
            '''
            can be called in main.py, useful for training resume.
            '''

        def cur_lr(self):
            '''
            return current learning rate.
            '''
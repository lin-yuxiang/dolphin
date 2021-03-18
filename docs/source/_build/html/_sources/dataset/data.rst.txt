.. _submodule_data:

Submodule of Dataset
====================

Modules of dataset, inherit from *torch.utils.data.Dataset*, which is an iteration
class provides batches of data to model.

A custom dataset module can be also easily used by adding two lines of code:

.. code-block:: python

    from dolphin.utils import Registers

    @Registers.dataset.register
    class CustomDataset():

        def __init__(self):
            ...


.. _pipeline:

Dataflow Pipeline
=================

There are more than 10 methods of data preprocessing methods implemented here, such as
image resizing, random image flip, random cropping and so on, which are design
with inheritance from **BaseModule**. All of them take a python dictionary as 
input and output with the same type. If customization of pipeline module is
needed, the new method must go by the same shape as said before.
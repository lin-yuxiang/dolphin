Module of Models
================

Normally a model of deep learning based computer vision algorithm is composed by
many parts, **backbone**, **neck**, **head** and so on. So their implementations
are stored in the directory with same name. The parent class of all model modules
is **BaseModelModule**.

The separation of directory **model** also follows the name of modules:

.. code-block:: text

   model/
      - algorithms/
      - modules/
         - backbone
         - neck
         - head
         - ...
      - utils

* :ref:`Submodule of Model <submodule_model>`
* :ref:`Algorithm <algorithm>`
* :ref:`Utility <utils_model>`

.. toctree::
   :maxdepth: 2
   :hidden:

   model.rst



:doc:`Back to Homepage </index>`
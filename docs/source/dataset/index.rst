Module of Dataset
=================

Module of Dataset plays a part in creating and preprocessing of dataset. It is
consist of directory *datasets* and *pipeline*. The part of *datasets* define
some entry of loading data information, and the other part of *pipeline* serves
as some preprocessing methods of dataset, such as image loading and augmentation.

The dataflow of dataset pipeline is implemented with a python **dictionary** which
carries plenty of dataset information. Every pipeline method takes the **dictionary**
as input and outputs corresponding **dictionary** includes processed dataset.
Hence, a new dataset module needed to be implemented as a iterator with 
such **dictionary** for using method of *pipeline*:

.. code-block:: python

   results = {
      'imgs (or imgs_file)': [],
      'label (or label_file)': [],
      'imgs_cfg': dict(),
      'label_cfg': dict()}

* :ref:`Submodule of Dataset <submodule_data>`
* :ref:`Dataflow Pipeline <pipeline>`


.. toctree::
   :maxdepth: 2
   :hidden:

   data.rst



:doc:`Back to Homepage </index>`
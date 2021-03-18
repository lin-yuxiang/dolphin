OpenDolphin Documentation
==========================

Welcome to project ``Dolphin``, which is an open source collection of algorithms 
using deep learning in several fields of computer vision based on **PyTorch**:
*Object Detection*, *Generative Adversarial Network*, *Video Action Analysis*, 
*Mono Depth Estimation*, *Activate Learning*, *Object Tracking* and 
*Segmentation*, aiming to simplifies algorithms research experiments.

.. note::
   This documentation contains only API introduction of the project, the 
   installation guidance please check our repo.

Wide Coverage
   A variety of computer vision algorithms are integrated, for every included 
   field of them, there exists at least one specific algorithm:

   * **Object Detection**: `Faster RCNN <https://arxiv.org/abs/1506.01497>`_
   * **Video Action Analysis**: `SlowFast <https://arxiv.org/abs/1812.03982>`_ for action recognition, `BMN <https://arxiv.org/abs/1907.09702>`_ for temporal action localization, `MOC-Detector <https://arxiv.org/abs/2001.04608>`_ for temporal and spatial action detection 
   * **Generative Adversarial Network**: `CycleGAN <https://arxiv.org/abs/1703.10593>`_
   * **Mono Depth Estimation**: `FCRN <https://arxiv.org/abs/1606.00373>`_
   * **Activate Learning**: `Query-based Entropy Sampling <https://ieeexplore.ieee.org/document/6889457>`_
   * **Object Tracking**: `FairMOT <https://arxiv.org/abs/2004.01888>`_
   * **Segmentation**: `FCN <https://arxiv.org/abs/1411.4038>`_

Modular Design
   The workflow of algorithm is separated into several modules: dataset 
   establishing, data augmentation, model building and so on, that is convenient
   for customization and combination. What's more, all of setting of modules and
   hypeparameters can be easily done in a simply configuration file.

Flexible Engine
   For adapting to some special algorithms, such as GAN, Activate Learning, a
   flexible workflow engine is created. It is compatible with controlling 
   sequence of updating different models within a iteration (GAN) and special
   workflow phase (query-based activate learning algorithms).

The details about this system are provided as below, You can find out more 
in this documentation.

* :doc:`Module Configuration </configuration/index>`
* :doc:`Module of Model </model/index>`
* :doc:`Module of Dataset </dataset/index>`
* :doc:`Module of Engine </engine/index>`

.. toctree::
   :maxdepth: 1
   :caption: OpenDolphin Documentation
   :hidden:

   /configuration/index.rst
   /model/index.rst
   /dataset/index.rst
   /engine/index.rst

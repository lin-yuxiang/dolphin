Module Configuration
====================

Algorithms within this project are divided into four main modules: **model**,
**loss**, **dataset** and **engine**, that are managed by *Register* mechanics.
For convenient, the importing and configuration of modules, parameters are 
format as *yaml* files.

Similarly, the directory structure of system is arranged as different modules
shown as below, for orderly management. Hence, in the documentation they would
be introduced one by one according to the structure. Additionally, some modules 
include submodules, they are presented in corresponding chapter.

.. code-block:: text

   dolphin/
      - main.py
      - configs/
      - dataset/
      - model/
      - loss/
      - engine/
      - utils/

The **main.py** file is entry of program, which is capable to parse 
configuration from *yaml* file, import the assigned modules, start the engine
and so on. That is to say, parameters of 

To learn more about configuration file and the *Register* mechanics for modules 
management, take a look at the sections described below.

* :ref:`Hierarchy of Configuration File <conf_file>`
* :ref:`Management of Modules <management>`

.. toctree::
   :maxdepth: 2
   :hidden:

   conf.rst

:doc:`Back to Homepage </index>`
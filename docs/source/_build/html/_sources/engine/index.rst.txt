Module of Engine
================

The usage of *engine* is to collect all modules (data, model, algorithm and so on)
together, build an whole pipeline, then organize them orderly to run with 
desired phase (normally training, validation or testing). 
So the module of engine possesses methods correspondingly.

.. note::
   The name of assigned phase in *config file* must be similar to name of the 
   corresponding method in *engine module*. For example, if you specify a process
   phase name 'train' in *config file*, your *engine module* must have the method
   with name of 'train'.

This design is compatible with special mode of algorithms like the query-based activate
learning, it needs the phase of query between training and validation. In summary,
if a new method are necessary to be called as a process phase, it's flexible to
firstly implement it within class engine, then assign its name in *config file* 
for use.

The directory of *engine* is separated with three main parts: Solver, Base Engines
and Work Flow Mix-in Methods. Details about them can be read in sub chapters below.

* :ref:`Solver <solver>`
* :ref:`Base Engine <base_engine>`
* :ref:`Work Flow Mix-in methods <mixin>`

.. toctree::
   :maxdepth: 2
   :hidden:

   engine.rst

:doc:`Back to Homepage </index>`
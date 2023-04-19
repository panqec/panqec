PanQEC
==================================

PanQEC is a Python package for simulating and visualizing quantum error correcting codes.
It allows one to easily create quantum codes, error models and decoders, and evaluate their performance
by calculating performance metrics, such as the threshold and subthreshold error rates.
Once a 2D or 3D quantum code is created, it can be visualized and manipulated on a web interface,
helping to verify the correct implementation of a code or understand the behavior of a decoder

To make a feature request or report a bug, please visit the PanQEC `GitHub repository <https://github.com/ehua7365/bn3d>`_.

See this `paper on the arXiv <https://arxiv.org/abs/2211.02116>`_
for more information about some of these codes.


.. image:: https://user-images.githubusercontent.com/1157968/180657086-48ea0da0-6da4-4f9c-88e5-4a3f1233db40.png

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   development

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/Panqec basics
   tutorials/Adding new code
   tutorials/Computing threshold
   tutorials/Running experiments on a cluster
   tutorials/XZZX threshold vs bias

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   cli
   codes
   error_models
   decoders
   simulation
   analysis
   misc


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

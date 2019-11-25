.. Etaler documentation master file, created by
   sphinx-quickstart on Mon Nov 25 14:13:56 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Etaler's documentation!
==================================

Etaler in an high-performance implementation of Numenta's HTM algorithms in C++.
It is diesigned to be used in real world applications and research projects.

Etaler provides:

* HTM algorithms with modern API
* A minimal cross-platform (CPU, GPU, etc..) Tensor implementation

Etaker requires a modern C++ compiler supporting C++17. The following C++ compilers are supported

* On Windows, Visual C++2019 or better
* On *nix systems, GCC 8.2 or a recent version of clang

Licensing
==================================

Etaler is licensed under BSD 3-Clause License. So use it freely!

Be aware that Numenta holds the rights to HTM related patents. And only allows free (as "free beers" free) use of their patents for non-commercial purpose. If you are using Etaler commercially; please contact Numenta for licensing.
(tl;dr Etaler is free for any purpose. But HTM is not for commercial use.)

.. toctree::
   :maxdepth: 1
   :caption: INSTALLATION

   BuildOnMSVC
   BuildOnOSX

.. toctree::
   :maxdepth: 2
   :caption: USAGE

   Introduction
   Tensor
   PythonBindings
   UsingWithClingROOT
   Contribution
   GUI

.. toctree::
   :caption: DEVELOPER ZONE
   :maxdepth: 2

   Backends
   DeveloperNotes
   OpenCLBackend


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Etaler:: https://github.com/etaler/Etaler
.. _Numenta:: https://numenta.com/

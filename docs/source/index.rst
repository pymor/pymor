#########################################
pyMOR - Model Order Reduction with python
#########################################

`pyMOR <https://pymor.org>`_ is a software library for building model order
reduction applications with the Python programming language. Implemented
algorithms include reduced basis methods for parametric linear and non-linear
problems, as well as system-theoretic methods such as balanced truncation or
IRKA.  All algorithms in pyMOR are formulated in terms of abstract interfaces
for seamless integration with external PDE solver packages.  Moreover, pure
Python implementations of finite element and finite volume discretizations
using the `NumPy/SciPy <https://scipy.org>`_ scientific computing stack are
provided for getting started quickly.


.. toctree::
    getting_started
    technical_overview
    environment
    release_notes
    bibliography
    notebooks

API Documentation
*****************

.. toctree::
    :maxdepth: 3
    :glob:

    generated/pymor

Demo Applications
*****************

.. toctree::
    :maxdepth: 3
    :glob:

    generated/pymordemos

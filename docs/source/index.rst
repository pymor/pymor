#########################################
pyMOR - Model Order Reduction with python
#########################################

`pyMOR <http://pymor.org>`_ is a software library for building model order
reduction applications with the Python programming language.  Its main focus
lies on the application of reduced basis methods to parameterized partial
differential equations.  All algorithms in pyMOR are formulated in terms of
abstract interfaces for seamless integration with external high-dimensional PDE
solvers.  Moreover, pure Python implementations of finite element and finite
volume discretizations using the `NumPy/SciPy <http://scipy.org>`_ scientific
computing stack are provided for getting started quickly.


.. toctree::
    getting_started
    technical_overview
    environment
    release_notes

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

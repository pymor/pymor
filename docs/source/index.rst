#########################################
pyMOR - Model Order Reduction with python
#########################################

`pyMOR <http://pymor.org>`_ is a software library developed at the University
of Münster for building model order reduction applications with the Python
programming language.  Its main focus lies on the reduction of parameterized
partial differential equations using the reduced basis method.  All algorithms
in pyMOR are formulated in terms of abstract interfaces for seamless
integration with external high-dimensional PDE-solver. Moreover, pure Python
implementations of finite element and finite volume discretizations using the
`NumPy/SciPy <http://scipy.org>`_ scientific computing stack are provided for
quick and easy prototyping.

.. toctree::
    getting_started
    technical_overview
    environment

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

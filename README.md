![pyMOR Logo](./logo/pymor_logo.svg)

pyMOR - Model Order Reduction with Python
=========================================

pyMOR is a software library for building model order
reduction applications with the Python programming language. Implemented
algorithms include reduced basis methods for parametric linear and non-linear
problems, as well as system-theoretic methods such as balanced truncation or
IRKA (Iterative Rational Krylov Algorithm).  All algorithms in pyMOR are
formulated in terms of abstract interfaces for seamless integration with
external PDE (Partial Differential Equation) solver packages.  Moreover, pure
Python implementations of FEM (Finite Element Method) and FVM (Finite Volume
Method) discretizations using the NumPy/SciPy scientific computing stack are
provided for getting started quickly.

[![PyPI](https://img.shields.io/pypi/pyversions/pymor.svg)](https://pypi.python.org/pypi/pymor)
[![PyPI](https://img.shields.io/pypi/v/pymor.svg)](https://pypi.python.org/pypi/pymor)
[![Docs](https://img.shields.io/endpoint?url=https%3A%2F%2Fdocs.pymor.org%2Fbadge.json)](https://docs.pymor.org/)
[![DOI](https://zenodo.org/badge/9220688.svg)](https://zenodo.org/badge/latestdoi/9220688)
[![GitLab Pipeline](https://zivgitlab.uni-muenster.de/pymor/pymor/badges/main/pipeline.svg)](https://zivgitlab.uni-muenster.de/pymor/pymor/commits/main)
[![Conda Tests](https://github.com/pymor/pymor/actions/workflows/conda_tests.yml/badge.svg)](https://github.com/pymor/pymor/actions/workflows/conda_tests.yml)
[![codecov](https://codecov.io/gh/pymor/pymor/branch/main/graph/badge.svg)](https://codecov.io/gh/pymor/pymor)

License
-------

Copyright pyMOR developers and contributors. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following
  disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The following files contain source code originating from other open source software projects:

* docs/source/pymordocstring.py  (sphinxcontrib-napoleon)
* src/pymor/algorithms/genericsolvers.py (SciPy)

See these files for more information.

Citing
------

If you use pyMOR for academic work, please consider citing our
[publication](https://doi.org/10.1137/15M1026614):

    R. Milk, S. Rave, F. Schindler
    pyMOR - Generic Algorithms and Interfaces for Model Order Reduction
    SIAM J. Sci. Comput., 38(5), pp. S194--S216, 2016

Installation via pip
--------------------

We recommend installation of pyMOR in a [virtual environment](https://virtualenv.pypa.io/en/latest/).

pyMOR can easily be installed with the [pip](https://pip.pypa.io/en/stable/)
command:

    pip install --upgrade pip  # make sure that pip is reasonably new
    pip install pymor[full]

(Please note that pip versions prior to 21.1 might have problems resolving all dependencies)

This will install the latest release of pyMOR on your system with most optional
dependencies.
For Linux we provide binary wheels, so no further system packages should
be required. Use

    pip install pymor

for an installation with minimal dependencies.
There are some optional packages not included with `pymor[full]`
because they need additional setup on your system:

* for support of MPI distributed models and parallelization of greedy algorithms
  (requires MPI development headers and a C compiler):

      pip install mpi4py

* dense matrix equation solver for system-theoretic MOR methods, required for
  H-infinity norm calculation (requires OpenBLAS headers and a Fortran
  compiler):

      pip install slycot

* dense and sparse matrix equation solver for system-theoretic MOR methods
  (other backends available):
  * from [source](https://gitlab.mpi-magdeburg.mpg.de/mess/cmess-releases)
    (recommended)
  * using a [wheel](https://www.mpi-magdeburg.mpg.de/projects/mess)

If you are not operating in a virtual environment, you can pass the optional `--user`
argument to pip. pyMOR will then only be installed for your
local user, not requiring administrator privileges.

To install the latest development version of pyMOR, execute

    pip install git+https://github.com/pymor/pymor#egg=pymor[full]

which will require that the [git](https://git-scm.com/) version control system is
installed on your system.

From time to time, the main branch of pyMOR undergoes major changes and things
might break (this is usually announced in our [discussion forum](https://github.com/pymor/pymor/discussions)),
so you might prefer to install pyMOR from the current release branch:

    pip install git+https://github.com/pymor/pymor@2022.2.x#egg=pymor[full]

Release branches will always stay stable and will only receive bugfix commits
after the corresponding release has been made.

Installation via conda
----------------------

pyMOR can be installed using `conda` by running

    conda install -c conda-forge pymor

Documentation
-------------

Documentation is available [online](https://docs.pymor.org/)
or you can build it yourself from inside the root directory of the pyMOR source tree
by executing:

    make docs

This will generate HTML documentation in `docs/_build/html`.

Useful Links
------------

* [Latest Changelog](https://docs.pymor.org/latest/release_notes/all.html)
* [Getting Started](https://docs.pymor.org/latest/getting_started.html)
* [Dependencies](https://github.com/pymor/pymor/blob/2022.2.x/requirements.txt)

External PDE solvers
--------------------

pyMOR has been designed with easy integration of external PDE solvers
in mind.

A basic approach is to use the solver only to generate high-dimensional
system matrices which are then read by pyMOR from disk (`pymor.discretizers.disk`).
Another possibility is to steer the solver via an appropriate network
protocol.

Whenever possible, we recommend to recompile the solver as a
Python extension module which gives pyMOR direct access to the solver without
any communication overhead. A basic example using
[pybind11](https://github.com/pybind/pybind11) can be found in
`src/pymordemos/minimal_cpp_demo`. Moreover,
we provide bindings for the following solver libraries:

* [FEniCS](https://fenicsproject.org)

    MPI-compatible wrapper classes for dolfin linear algebra data structures are
    shipped with pyMOR (`pymor.bindings.fenics`).
    For an example see `pymordemos.thermalblock`, `pymordemos.thermalblock_simple`.
    It is tested using FEniCS version 2019.1.0.

* [deal.II](https://dealii.org)

    Python bindings and pyMOR wrapper classes can be found
    [here](https://github.com/pymor/pymor-deal.II).

* [NGSolve](https://ngsolve.org)

    Wrapper classes for the NGSolve finite element library are shipped with pyMOR
    (`pymor.bindings.ngsolve`).
    For an example see `pymordemos.thermalblock_simple`.
    It is tested using NGSolve version v6.2.2104.

Do not hesitate to contact
[us](https://github.com/pymor/pymor/discussions) if you
need help with the integration of your PDE solver.

External Matrix Equation Solvers
--------------------------------

pyMOR also provides bindings to matrix equation solvers (in `pymor.bindings`),
which are needed for the system-theoretic methods and need to be installed
separately. Bindings for the following solver libraries are included:

* [Py-M.E.S.S.](https://www.mpi-magdeburg.mpg.de/projects/mess)

    The Matrix Equation Sparse Solver library is intended for solving large sparse matrix equations (`pymor.bindings.pymess`).

* [Slycot](https://github.com/python-control/Slycot)

    Python wrapper for the Subroutine Library in Systems and Control Theory (SLICOT) is also used for Hardy norm computations (`pymor.bindings.slycot`).

Environments for pyMOR Development and Tests
-----------------------------------------------

Please see the [Developer Documentation](https://docs.pymor.org/latest/developer_docs.html).

Contact
-------

Should you have any questions regarding pyMOR or wish to contribute,
do not hesitate to contact us via our GitHub discussions forum:

<https://github.com/pymor/pymor/discussions>

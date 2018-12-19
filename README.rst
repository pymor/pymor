pyMOR - Model Order Reduction with Python
=========================================

pyMOR is a software library for building model order reduction
applications with the Python programming language. Implemented
algorithms include reduced basis methods for parametric linear and
non-linear problems, as well as system-theoretic methods such as
balanced truncation or IRKA. All algorithms in pyMOR are formulated in
terms of abstract interfaces for seamless integration with external PDE
solver packages. Moreover, pure Python implementations of finite element
and finite volume discretizations using the NumPy/SciPy scientific
computing stack are provided for getting started quickly.

`Latest Docs <http://pymor.readthedocs.org/en/latest>`__
`DOI <https://zenodo.org/badge/latestdoi/9220688>`__ `Build
Status <https://travis-ci.org/pymor/pymor>`__

License
-------

Copyright 2013-2018 pyMOR developers and contributors. All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

-  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
-  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The following files contain source code originating from other open
source software projects:

-  docs/source/pymordocstring.py (sphinxcontrib-napoleon)
-  src/pymor/algorithms/genericsolvers.py (SciPy)

See these files for more information.

Citing
------

If you use pyMOR for academic work, please consider citing our
`publication <https://epubs.siam.org/doi/abs/10.1137/15M1026614>`__:

::

   R. Milk, S. Rave, F. Schindler
   pyMOR - Generic Algorithms and Interfaces for Model Order Reduction
   SIAM J. Sci. Comput., 38(5), pp. S194-S216

Installation via pip
--------------------

We recommend installation of pyMOR in a `virtual
environment <https://virtualenv.pypa.io/en/latest/>>`__.

pyMOR can easily be installed with the
`pip <https://pip.pypa.io/en/stable/>`__ command:

::

   pip install --upgrade pip  # make sure that pip is reasonably new
   pip install pymor[full]

This will install the latest release of pyMOR on your system with most
optional dependencies. For Linux we provide binary wheels, so no further
system packages should be required. Use

::

   pip install pymor

for an installation with minimal dependencies. If you are not operating
in a virtual environment, you can pass the optional ``--user`` argument
to pip, pyMOR will then only be installed for your local user, not
requiring administrator privileges. To install the latest development
version of pyMOR, execute

::

   pip install git+https://github.com/pymor/pymor#egg=pymor[full]

which will require that the `git <https://git-scm.com/>`__ version
control system is installed on your system.

From time to time, the master branch of pyMOR undergoes major changes
and things might break (this is usually announced on our `mailing
list <http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev>`__),
so you might prefer to install pyMOR from the current release branch:

::

   pip install git+https://github.com/pymor/pymor@0.5.x#egg=pymor[full]

Release branches will always stay stable and will only receive bugfix
commits after the corresponding release has been made.

Documentation
-------------

Documentation is available online at `Read the
Docs <https://pymor.readthedocs.org/>`__ or you can build it yourself
from inside the root directory of the pyMOR source tree by executing:

::

   make doc

This will generate HTML documentation in ``docs/_build/html``.

External PDE solvers
--------------------

pyMOR has been designed with easy integration of external PDE solvers in
mind.

A basic approach is to use the solver only to generate high-dimensional
system matrices which are then read by pyMOR from disk
(``pymor.discretizers.disk``). Another possibility is to steer the
solver via an appropriate network protocol.

Whenever possible, we recommend to recompile the solver as a Python
extension module which gives pyMOR direct access to the solver without
any communication overhead. A basic example using
`pybindgen <https://github.com/gjcarneiro/pybindgen>`__ can be found in
``src/pymordemos/minimal_cpp_demo``. A more elaborate nonlinear example
using `Boost.Python <http://www.boost.org/>`__ can be found
`here <https://github.com/pymor/dune-burgers-demo>`__. Moreover, we
provide bindings for the following solver libraries:

-  `FEniCS <http://fenicsproject.org>`__

   MPI-compatible wrapper classes for dolfin linear algebra data
   structures are shipped with pyMOR (``pymor.bindings.fenics``). For an
   example see ``pymordemos.thermalbock``,
   ``pymordemos.thermalblock_simple``.

-  `deal.II <https://dealii.org>`__

   Python bindings and pyMOR wrapper classes can be found
   `here <https://github.com/pymor/pymor-deal.II>`__.

-  `NGSolve <https://ngsolve.org>`__

   Wrapper classes for the NGSolve finite element library are shipped
   with pyMOR (``pymor.bindings.ngsolve``). For an example see
   ``pymordemos.thermalblock_simple``.

Do not hesitate to contact
`us <http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev>`__ if
you need help with the integration of your PDE solver.

Setting up an Environment for pyMOR Development
-----------------------------------------------

First make sure that all dependencies are installed. This can be easily
achieved by first installing pyMOR with its dependencies as described
above. Then uninstall the pyMOR package itself, e.g.

::

   pip uninstall pyMOR

Then, clone the pyMOR git repository using

::

   git clone https://github.com/pymor/pymor $PYMOR_SOURCE_DIR
   cd $PYMOR_SOURCE_DIR

and, optionally, switch to the branch you are interested in, e.g.

::

   git checkout 0.5.x

Then, add pyMOR to the search path of your Python interpreter, either by
setting PYTHONPATH

::

   export PYTHONPATH=$PYMOR_SOURCE_DIR/src:$PYTHONPATH

or by using a .pth file:

::

   echo "$PYMOR_SOURCE_DIR/src" > $PYTHON_ROOT/lib/python3.6/site-packages/pymor.pth

Here, PYTHON_ROOT is either '/usr', '$HOME/.local' or the root of your
`virtual environment <http://www.virtualenv.org/>`__. Finally, build the
Cython extension modules as described in the next section.

Cython extension modules
------------------------

pyMOR uses `Cython <http://www.cython.org/>`__ extension modules to
speed up numerical algorithms which cannot be efficiently expressed
using NumPy idioms. The source files of these modules (files with
extension ``.pyx``) have to be processed by Cython into a ``.c``-file
which then must be compiled into a shared object (``.so`` file). The
whole build process is handeled automatically by ``setup.py``.

If you want to develop Cython extensions modules for pyMOR yourself, you
should add your module to the ``ext_modules`` list defined in the
``_setup`` method of ``setup.py``. Calling

::

   python setup.py build_ext --inplace

will then build the extension module and place it into your pyMOR source
tree.

Tests
-----

pyMOR uses `pytest <https://pytest.org/>`__ for unit testing. To run the
test suite, simply execute ``make test`` in the base directory of the
pyMOR repository. This will also create a test coverage report which can
be found in the ``htmlcov`` directory. Alternatively, you can run
``make full-test`` which will also enable
`pyflakes <https://pypi.python.org/pypi/pyflakes>`__ and
`pep8 <https://www.python.org/dev/peps/pep-0008/>`__ checks.

All tests are contained within the ``src/pymortests`` directory and can
be run individually by executing
``py.test src/pymortests/the_module.py``.

Contact
-------

Should you have any questions regarding pyMOR or wish to contribute, do
not hestitate to contact us via our development mailing list:

http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev

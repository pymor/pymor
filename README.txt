pyMOR - Model Order Reduction with Python
=========================================

pyMOR is a software library for building model order reduction
applications with the Python programming language. Its main focus lies
on the application of reduced basis methods to parameterized partial
differential equations. All algorithms in pyMOR are formulated in terms
of abstract interfaces for seamless integration with external
high-dimensional PDE solvers. Moreover, pure Python implementations of
finite element and finite volume discretizations using the NumPy/SciPy
scientific computing stack are provided for getting started quickly.

[Docs] [Docs] [DOI] [Build Status] [Coverage Status]

License
-------

Copyright 2013-2016 pyMOR developers and contributors. All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

-   Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
-   Redistributions in binary form must reproduce the above copyright
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

-   docs/source/pymordocstring.py (sphinxcontrib-napoleon)
-   src/pymor/la/genericsolvers.py (SciPy)

See these files for more information.

Distribution Packages
---------------------

Packages for Ubuntu are available via our PPA:

    sudo apt-add-repository ppa:pymor/stable
    sudo apt-get update
    sudo apt-get install python-pymor

Daily snapshots are available via the pymor/daily PPA.

Demo applications and documentation are packaged separately:

    sudo apt-get install python-pymor-demos
    sudo apt-get install python-pymor-doc

The latter makes a pymor-demo script available, which can be used to run
all installed demos.

Installation via pip
--------------------

pyMOR can also easily be installed via the pip command:

    pip install [--user] pymor

This will install the latest release of pyMOR on your system. If you
pass the optional --user argument, pyMOR will only be installed for your
local user, not requiring administrator privileges. To install the
latest development version of pyMOR, execute

    pip install [--user] git+https://github.com/pymor/pymor

which will require that the git version control system is installed on
your system.

From time to time, the master branch of pyMOR undergoes major changes
and things might break (this is usually announced on our mailing list),
so you might prefer to install pyMOR from the current release branch:

    pip install [--user] git+https://github.com/pymor/pymor@0.4.x

Release branches will always stay stable and will only receive bugfix
commits after the corresponding release has been made.

Note that pyMOR depends on Cython, as well as the NumPy and SciPy
packages. On all major Linux distributions, these packages can be easily
installed via the distribution's package manager. For Debian-based
systems (e.g. Ubuntu), the following command should work:

    sudo apt-get install cython python-pip python-numpy python-scipy

When not available on your system, pip will automatically build and
install these dependencies. This, however, will in turn require a full
C/C++ compiler toolchain and header files for several libraries (BLAS,
etc.).

After installation of pyMOR, further optional packages will be suggested
if not already installed. Some of these (PySide, matplotlib, pyopengl,
mpi4py) are again most easily installed via your package manager. For
Debian-based systems, try:

    sudo apt-get install python-pyside python-matplotlib python-opengl python-mpi4py

Again, all these dependencies can also be installed directly via pip.

Documentation
-------------

Documentation is available online at Read the Docs or offline in the
python-pymor-doc package.

To build the documentation yourself, execute

    make doc

inside the root directory of the pyMOR source tree. This will generate
HTML documentation in docs/_build/html.

External PDE solvers
--------------------

pyMOR has been designed with easy integration of external PDE solvers in
mind.

A basic approach is to use the solver only to generate high-dimensional
system matrices which are then read by pyMOR from disk
(pymor.discretizers.disk). Another possibility is to steer the solver
via an appropriate network protocol.

Whenever possible, we recommend to recompile the solver as a Python
extension module which gives pyMOR direct access to the solver without
any communication overhead. A basic example using pybindgen can be found
in src/pymordemos/minimal_cpp_demo. A more elaborate nonlinear example
using Boost.Python can be found here. Moreover, we provide bindings for
the following solver libraries:

-   FEniCS

    MPI-compatible wrapper classes for dolfin linear algebra data
    structures are shipped with pyMOR (pymor.vectorarrays.fenics,
    pymor.operators.fenics). For an example see pymordemos.thermalbock,
    pymordemos.thermalblock_simple.

-   deal.II

    Python bindings and pyMOR wrapper classes can be found here.

-   DUNE

    dune-pymor automatically wraps dune-hdd discretizations for use with
    pyMOR.

Do not hesitate to contact us if you need help with the integration of
your PDE solver.

Setting up an Environment for pyMOR Development
-----------------------------------------------

First make sure that all dependencies are installed. This can be easily
achieved by first installing pyMOR with its dependencies as described
above. Then uninstall the pyMOR package itself, e.g.

    sudo apt-get uninstall python-pymor

or

    pip uninstall pyMOR

Then, clone the pyMOR git repository using

    git clone https://github.com/pymor/pymor $PYMOR_SOURCE_DIR
    cd $PYMOR_SOURCE_DIR

and, optionally, switch to the branch you are interested in, e.g.

    git checkout 0.4.x

Then, add pyMOR to the search path of your Python interpreter, either by
setting PYTHONPATH

    export PYTHONPATH=$PYMOR_SOURCE_DIR/src:$PYTHONPATH

or by using a .pth file:

    echo "$PYMOR_SOURCE_DIR/src" > $PYTHON_ROOT/lib/python2.7/site-packages/pymor.pth

Here, PYTHON_ROOT is either '/usr', '$HOME/.local' or the root of your
virtual environment. Finally, build the Cython extension modules as
described in the next section.

Cython extension modules
------------------------

pyMOR uses Cython extension modules to speed up numerical algorithms
which cannot be efficiently expressed using NumPy idioms. The source
files of these modules (files with extension .pyx) have to be processed
by Cython into a .c-file which then must be compiled into a shared
object (.so file). The whole build process is handeled automatically by
setup.py.

If you want to develop Cython extensions modules for pyMOR yourself, you
should add your module to the ext_modules list defined in the _setup
method of setup.py. Calling

    python setup.py build_ext --inplace

will then build the extension module and place it into your pyMOR source
tree.

Tests
-----

pyMOR uses pytest for unit testing. To run the test suite, simply
execute make test in the base directory of the pyMOR repository. This
will also create a test coverage report which can be found in the
htmlcov directory. Alternatively, you can run make full-test which will
also enable pyflakes and pep8 checks.

All tests are contained within the src/pymortests directory and can be
run individually by executing py.test src/pymortests/the_module.py.

Contact
-------

Should you have any questions regarding pyMOR or wish to contribute, do
not hestitate to contact us via our development mailing list:

http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev


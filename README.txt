pyMOR - Model Order Reduction with Python
=========================================

pyMOR is a software library developed at the University of Münster for
building model order reduction applications with the Python programming
language. Its main focus lies on the reduction of parameterized partial
differential equations using the reduced basis method. All algorithms in
pyMOR are formulated in terms of abstract interfaces for seamless
integration with external high-dimensional PDE-solver. Moreover, pure
Python implementations of finite element and finite volume
discretizations using the NumPy/SciPy scientific computing stack are
provided for getting started quickly.

NOTE pyMOR is still in early development. Should you have any questions
regarding pyMOR or wish to contribute, do not hesitate to contact us!

[Build Status]

License
-------

Copyright (c) 2013, 2014, Rene Milk, Stephan Rave, Felix Schindler All
rights reserved.

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

Distribution Packages
---------------------

Packages for Ubuntu 12.04 and 13.10 are available via our PPA:

    sudo apt-add-repository ppa:pymor/stable
    sudo apt-get update
    sudo apt-get install python-pymor

Demo applications and documentation are packaged separately:

    sudo apt-get install python-pymor-demos
    sudo apt-get install python-pymor-doc

The latter makes a pymor-demo script available, which can be used to run
all installed demos.

Installation into a virtualenv
------------------------------

When installing pyMOR manually, we recommend installation into a
dedicated Python virtualenv. On Debian based systems, install virtualenv
using

    sudo apt-get install python-virtualenv

On Ubuntu systems, you may also wish to install pyMOR's dependencies
system-wide using

    sudo apt-add-repository ppa:pymor/stable
    sudo apt-get update
    sudo apt-get build-dep python-pymor

Then create a new virtualenv and activate it:

    virtualenv --system-site-packages $PATH_TO_VIRTUALENV
    source $PATH_TO_VIRTUALENV/bin/activate

The --system-site-packages flag makes Python packages installed by your
distribution available inside the virtualenv. If you do not wish this
behaviour, simply remove the flag.

On older distributions you will have to upgrade the distribute package.
Moreover, if NumPy and Cython are not already available in the
virtualenv, we will have to install them manually. (Automatic dependency
resolution via pip fails for these packages. To build NumPy and, later,
SciPy, you will need to have Fortran as well as BLAS and LAPACK headers
installed on your system.)

    pip install --upgrade distribute
    pip install cython
    pip install numpy

Finally install pyMOR itself with all missing dependencies:

    pip install pymor

The installation script might recommend the installation of additional
packages. (This is easy to miss, as pip will install dependencies after
pyMOR itself has been installed, so search at the top of your console
log!) You will most likely want to install IPython and, in particular,
matplotlib, PyOpenGL, glumpy, and PySide. The latter packages are
required for pyMOR's visualization routines. At the time of writing,
PySide fails to install with current Sphinx versions, so we have to
install an old one:

    pip install "sphinx==1.1.3"
    pip install ipython pyside matplotlib pyopengl glumpy 

Documentation
-------------

Documentation is available online at Read the Docs or offline in the
python-pymor-doc package.

To build the documentation yourself, execute

    make doc

inside the root directory of the pyMOR source tree. This will generate
HTML documentation in docs/_build/html.

Setting up an Environment for pyMOR Development
-----------------------------------------------

If you want to modify (or extend!) pyMOR itself, we recommend to setup a
virtualenv for development (see above). The virtualenv should have all
dependencies of pyMOR available. On Ubuntu machines, you can simply
install pyMOR from our PPA and then create an empty virtualenv with
system site-packages enabled. Otherwise, follow the above instructions
for installing pyMOR inside a virtualenv. However, pyMOR itself should
not be installed inside the virtualenv. If it is, use

    pip uninstall pymor

to remove it. Then, clone the pyMOR git repository using

    git clone https://github.com/pymor/pymor $PYMOR_SOURCE_DIR
    cd $PYMOR_SOURCE_DIR

and, optionally, switch to the branch you are interested in, e.g.

    git checkout 0.2.x

Finally add pyMOR to the path of your virtualenv:

    echo "$PYMOR_SOURCE_DIR" > $VIRTUAL_ENV/lib/python2.7/site-packages/pymor.pth

This will make pyMOR importable inside the virtualenv and will override
any other pyMOR versions installed on the system.

Cython extension modules
------------------------

pyMOR uses Cython extension modules to speed up numerical algorithms
which cannot be efficiently expressed using NumPy idioms. To benefit
from these optimizations, the modules' source files (currently
pymor/tools/inplace.pyx and pymor/tools/realations.pyx) have to be
processed by Cython into a .c-file which then must be compiled into a
shared object. These .so-files then take precedence over the
non-optimized pure python modules. The whole build process is handeled
automatically by setup.py.

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

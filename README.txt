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

Copyright (c) 2013, Felix Albrecht, Rene Milk, Stephan Rave  
All rights reserved.

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

Installation
------------

We recommend the following way to install pyMOR with all its
dependencies.

This procedure has been tested on the following platforms:

    Ubuntu 12.04, Ubuntu 13.04, Arch Linux

1.  Open a terminal and make sure you have git and python installed on
    your computer. (On Debian based linux distributions (e.g. Ubunutu)
    use sudo     apt-get install git python2.7 to install both git and
    python.)

2.  Clone the pyMOR main repository using

        git clone https://github.com/pymor/pymor.git

    This will create a directory named pymor in your current working
    directory. Change into this directory using

        cd pymor

3.  Excecute the provided installation script

        ./install.py

    The installation script will ensure that all necessary system
    libraries and development headers are installed, create a new python
    virtual environment (virtualenv, see www.virtualenv.org), install
    all necessary python packages into this virtualenv, and finally
    install pyMOR itself.

    The installation process can be customized using various
    command-line arguments. (See ./install.py --help.) Most notable, the
    path of the virtualenv can be configured using the --virtualenv-dir
    option. Moreover, if you intend to work directly inside the pyMOR
    source tree, use

        ./install.py --only-deps

    to prevent the installation of pyMOR itself into the site-packages
    directory of the virtualenv. In this case, the installation script
    will add the pyMOR source tree to the PYTHONPATH of the virtualenv,
    so pyMOR will always be importable inside the virtualenv. (This can
    be prevented by adding the --without-python-path option.) Moreover

        python setup.py build_ext --inplace

    will be automatically called, to build pyMOR's Cython extensions
    modules.

4.  Activate the new virtualenv. If you did not change the default path
    of the virtualenv, this can be done by executing

        source $HOME/virtualenv/pymor/bin/activate

5.  Try out one of the provided demos, e.g. call

        cd src/pymordemos
        ./thermalblock.py -ep --plot-solutions 2 2 3 16

Documentation
-------------

To build the documentation execute

    make doc

inside the root directory of the pyMOR source tree. This will generate
HTML documentation in 'docs/_build/html'. The documentation is also
available online on Read the Docs.

Cython extension modules
------------------------

pyMOR uses Cython extension modules to speed up numerical algorithms
which cannot be efficiently expressed using NumPy idioms. To benefit
from these optimizations, the modules' source files (currently
pymor/tools/inplace.pyx and pymor/tools/realations.pyx) have to be
processed by Cython into a .c-file which then must be compiled into a
shared object. These .so-files then take precedence over the
non-optimized pure python modules. This whole build process is handeled
automatically by setup.py which is internally called by the install.py
script.

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

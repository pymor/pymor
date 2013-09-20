pyMor - Model order reduction in Python
=======================================

**NOTE** pyMor is still alpha quality software and under heavy development.
Should you have any questions regarding pyMor or wish to contribute, do not
hesitate to directly contact one of the developers.

[![Build Status](https://travis-ci.org/pyMor/pyMor.png?branch=master)](https://travis-ci.org/pyMor/pyMor)


Installation
------------

We recommend the following way to install pyMor with all its dependencies.

This procedure has been tested on the following platforms:

    Ubuntu 12.04, Ubuntu 13.04, Arch Linux


1.  Open a terminal and make sure you have git and python installed on your
    computer. (On Debian based linux distributions (e.g. Ubunutu) use `sudo
    apt-get install git python2.7` to install both git and python.)

2.  Clone the pyMor main repository using
    
        git clone https://github.com/pyMor/pyMor.git
    
    This will create a directory named `pyMor` in your current working directory.
    Change into this directory using
    
        cd pyMor

3.  Excecute the provided installation script
    
        ./install.py
    
    The installation script will ensure that all necessary system libraries and
    development headers are installed, create a new python virtual environment
    (virtualenv, see www.virtualenv.org), install all necessary python packages into
    this virtualenv, and finally install pyMor itself.
    
    The installation process can be customized using various command-line arguments.
    (See `./install.py --help`.) Most notable, the path of the virtualenv can be
    configured using the `--virtualenv-dir` option.  Moreover, if you intend to work
    directly inside the pyMor source tree, use
    
        ./install.py --only-deps
    
    to prevent the installation of pyMor itself into the `site-packages` directory
    of the virtualenv. In this case, the installation script will add the pyMor
    source tree to the `PYTHONPATH` of the virtualenv, so pyMor will always be
    importable inside the virtualenv. (This can be prevented by adding the
    `--without-python-path` option.) Moreover
    
        python setup.py build_ext --inplace
    
    will be automatically called, to build pyMor's Cython extensions modules.
   
4.  Activate the new virtualenv. If you did not change the default path of the
    virtualenv, this can be done by executing
    
        source $HOME/virtualenv/pyMor/bin/activate
    
5.  Try out one of the provided demos, e.g. call
    
        cd src/pymor/demos
        ./thermalblock.py -ep --plot-solutions 2 2 3 16


Cython extension modules
------------------------

pyMor uses [Cython](http://www.cython.org/) extension modules to speed up
numerical algorithms which cannot be efficiently expressed using NumPy idioms.
To benefit from these optimizations, the modules' source files (currently
`pymor/tools/inplace.pyx` and `pymor/tools/realations.pyx`) have to be processed
by Cython into a `.c`-file which then must be compiled into a shared object.
These `.so`-files then take precedence over the non-optimized pure python
modules.  This whole build process is handeled automatically by `setup.py`
which is internally called by the `install.py` script.  

If you want to develop Cython extensions modules for pyMor yourself, you should
add your module to the `ext_modules` list defined in the `_setup` method of
`setup.py`. Calling

    python setup.py build_ext --inplace

will then build the extension module and place it into your pyMor source tree.


Debugging
---------

 * You can globally disable caching by having `PYMOR_CACHE_DISABLE=1` in the process' environment


Tests
-----

pyMor uses [pytest](http://pytest.org/) for unit testing. To run the test suite,
simply execute `make test` in the base directory of the pyMor repository. This
will also create a test coverage report which can be found in the `htmlcov`
directory. Alternatively, you can run `make full-test` which will also enable
[pyflakes](https://pypi.python.org/pypi/pyflakes) and
[pep8](http://www.python.org/dev/peps/pep-0008/) checks.

All tests are contained within the `src/pymortests` directory and can be run
individually by executing `py.test src/pymortests/the_module.py`.

pyMor - Model order reduction in Python
=======================================

**NOTE** pyMor is still alpha quality software and under heavy development.
Should you have any questions regarding pyMor or wish to contribute, do not
hesitate to directly contact one of the developers.


Installation
------------

We recommend the following way to install pyMor with all its dependencies.

This procedure has been tested on the following platforms:

    Ubuntu 12.04, Ubuntu 13.04, Arch Linux


1. Open a terminal and make sure you have git and python installed on your
   computer. (On Debian based linux distributions (e.g. Ubunutu) use `sudo
   apt-get install git python2.7` to install both git and python.)
   
2. Clone the pyMor main repository using

   ```
   git clone https://github.com/pyMor/pyMor.git
   ```

   This will create a directory named `pyMor` in your current working directory.
   Change into this directory using
   
   ```
   cd pyMor
   ```
   
3. Excecute the provided installation script
   
   ```
   ./install.sh
   ```
   
   The installation script will ensure that all necessary system libraries and
   development headers are installed, create a new python virtual environment
   (virtualenv, see www.virtualenv.org), install all necessary python packages into
   this virtualenv, and finally install pyMor itself.
   
   The installation process can be customized using various command-line arguments.
   (See `./install.sh --help`.) Most notable, the path of the virtualenv can be
   configured using the `--virtualenv-dir` option.  Moreover, if you intend to work
   directly inside the pyMor source tree, use
   
   ```
   ./install.sh --only-deps
   ```
   
   to prevent the installation of pyMor itself into the `site-packages` directory
   of the virtualenv. In this case, the installation script will add the pyMor
   source tree to the `PYTHONPATH` of the virtualenv, so pyMor will always be
   importable inside the virtualenv. (This can be prevented by adding the
   `--without-python-path` option.) Moreover
   
   ```
   python setup.py build_ext --inplace
   ```
   
   will be automatically called, to build pyMor's Cython extensions modules.
   
4. Activate the new virtualenv. If you did not change the default path of the
   virtualenv, this can be done by executing
   
   ```
   source $HOME/virtualenv/pyMor/bin/activate
   ```
   
5. Try out one of the provided demos, e.g. call
   
   ```
   cd src/pymor/demos
   ./thermalblock.py -ep --plot-solutions 2 2 3 16
   ```


Debugging
---------

 * You can globally disable caching by having `PYMOR_CACHE_DISABLE=1` in the process' environment


Tests
-----

You'll need mock, nose-cov, nose, nosehtmloutput, nose-progressive and tissue installed to run `make test`.
Having `PYMOR_NO_GRIDTESTS=1` in the process' environment disables all, expensive grid testing.

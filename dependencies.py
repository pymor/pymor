# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

tests_require = ['pytest', 'pytest-cov']
install_requires = ['cython>=0.20.1', 'numpy>=1.8.1', 'scipy>=0.13.3', 'Sphinx', 'docopt']
setup_requires = ['pytest-runner', 'cython>=0.20.1', 'numpy>=1.8.1']
install_suggests = {'ipython': 'an enhanced interactive python shell',
                    'ipyparallel': 'required for pymor.parallel.ipython',
                    'matplotlib': 'needed for error plots in demo scipts',
                    'pyopengl': 'fast solution visualization for builtin discretizations (PySide also required)',
                    'pyside': 'solution visualization for builtin discretizations',
                    'pyamg': 'algebraic multigrid solvers',
                    'mpi4py': 'required for pymor.tools.mpi and pymor.parallel.mpi',
                    'pytest': 'testing framework required to execute unit tests'}

import_names = {'ipython': 'IPython',
                'pytest-cache': 'pytest_cache',
                'pytest-capturelog': 'pytest_capturelog',
                'pytest-instafail': 'pytest_instafail',
                'pytest-xdist': 'xdist',
                'pytest-cov': 'pytest_cov',
                'pytest-flakes': 'pytest_flakes',
                'pytest-pep8': 'pytest_pep8',
                'pyopengl': 'OpenGL',
                'pyside': 'PySide'}

if __name__ == '__main__':
    print(' '.join([i for i in install_requires + install_suggests]))

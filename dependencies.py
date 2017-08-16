#!/usr/bin/env python3
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

tests_require = ['pytest>=3.0', 'pytest-cov']
install_requires = ['cython>=0.20.1', 'numpy>=1.8.1', 'scipy>=0.13.3', 'Sphinx>=1.4.0', 'docopt', 'Qt.py>=1.0.0b3']
setup_requires = ['pytest-runner>=2.9', 'cython>=0.20.1', 'numpy>=1.8.1']
install_suggests = {'ipython>=3.0': 'an enhanced interactive python shell',
                    'ipyparallel': 'required for pymor.parallel.ipython',
                    'matplotlib': 'needed for error plots in demo scipts',
                    'pyopengl': 'fast solution visualization for builtin discretizations (PySide also required)',
                    'pyamg': 'algebraic multigrid solvers',
                    'mpi4py': 'required for pymor.tools.mpi and pymor.parallel.mpi',
                    'https://bitbucket.org/pauloh/pyevtk/get/tip.tar.gz': 'writing vtk output',
                    'pytest>=3.0': 'testing framework required to execute unit tests',
                    'pyside; python_version < "3.5"': 'solution visualization for builtin discretizations',
                    'PyQt5 ; python_version >= "3.5"': 'solution visualization for builtin discretizations',
                    'pillow': 'image library used for bitmap data functions',
                    'psutil': 'Process management abstractions used for gui',
                    'slycot>=0.3.1': 'python wrapper for the SLICOT control and systems library'}

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
    print(' '.join([i for i in install_requires + list(install_suggests.keys())]))
    import os
    import itertools
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'wt') as req:
        for module in sorted(set(itertools.chain(install_requires, setup_requires))):
            req.write(module+'\n')
    with open(os.path.join(os.path.dirname(__file__), 'requirements-optional.txt'), 'wt') as req:
        req.write('-r requirements.txt\n')
        for module in sorted(set(itertools.chain(tests_require, install_suggests.keys()))):
            req.write(module+'\n')

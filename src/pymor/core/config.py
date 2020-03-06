# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from importlib import import_module
import sys
import platform
import warnings


def _can_import(module):
    try:
        import_module(module)
        return True
    except ImportError:
        pass
    return False


def _get_fenics_version():
    import dolfin as df
    if df.__version__ != '2019.1.0':
        warnings.warn(f'FEniCS bindings have been tested for version 2019.1.0 (installed: {df.__version__}).')
    return df.__version__


def is_windows_platform():
    return sys.platform == 'win32' or sys.platform == 'cygwin'


def is_macos_platform():
    return 'Darwin' in platform.system()


def _get_matplotib_version():
    import matplotlib
    if is_windows_platform():
        matplotlib.use('Qt4Agg')
    return matplotlib.__version__


def _get_ipython_version():
    try:
        import ipyparallel
        return ipyparallel.__version__
    except ImportError:
        import IPython.parallel
        return getattr(IPython.parallel, '__version__', True)


def _get_slycot_version():
    from slycot.version import version
    if list(map(int, version.split('.'))) < [0, 3, 1]:
        import warnings
        warnings.warn('Slycot support disabled (version 0.3.1 or higher required).')
        return False
    else:
        return version


def _get_qt_version():
    try:
        import Qt
        return Qt.__binding__ + ' ' + Qt.__binding_version__
    except AttributeError as ae:
        warnings.warn(f'importing Qt.py abstraction failed:\n{ae}')
        return False


def is_jupyter():
    """This Method is not foolprof and might fail with any given jupyter release
    :return: True if we believe to be running in a Jupyter Notebook or Lab
    """
    try:
        from IPython import get_ipython
    except (ImportError, ModuleNotFoundError):
        return False
    from os import environ
    force = environ.get('PYMOR_FORCE_JUPYTER', None)
    if force is not None:
        return bool(force)
    return type(get_ipython()).__module__.startswith('ipykernel.')


def is_nbconvert():
    """In some visualization cases we need to be able to detect if a notebook
    is executed with nbconvert to disable async loading
    """
    from os import environ
    return is_jupyter() and bool(environ.get('PYMOR_NBCONVERT', False))


_PACKAGES = {
    'CYTHON': lambda: import_module('cython').__version__,
    'DEALII': lambda: import_module('pydealii'),
    'DOCOPT': lambda: import_module('docopt').__version__,
    'FENICS': _get_fenics_version,
    'GL': lambda: import_module('OpenGL.GL') and import_module('OpenGL').__version__,
    'IPYTHON': _get_ipython_version,
    'MATPLOTLIB': _get_matplotib_version,
    'MESHIO': lambda: import_module('meshio').__version__,
    'IPYWIDGETS': lambda: import_module('ipywidgets').__version__,
    'MPI': lambda: import_module('mpi4py.MPI') and import_module('mpi4py').__version__,
    'NGSOLVE': lambda: bool(import_module('ngsolve')),
    'NUMPY': lambda: import_module('numpy').__version__,
    'PYAMG': lambda: import_module('pyamg.version').full_version,
    'PYMESS': lambda: bool(import_module('pymess')),
    'PYTEST': lambda: import_module('pytest').__version__,
    'PYTHREEJS': lambda: import_module('pythreejs._version').__version__,
    'PYEVTK': lambda: _can_import('pyevtk'),
    'QT': _get_qt_version,
    'QTOPENGL': lambda: bool(import_module('Qt.QtOpenGL')),
    'SCIPY': lambda: import_module('scipy').__version__,
    'SCIPY_LSMR': lambda: hasattr(import_module('scipy.sparse.linalg'), 'lsmr'),
    'SLYCOT': lambda: _get_slycot_version(),
    'SPHINX': lambda: import_module('sphinx').__version__,
}


class Config:

    def __init__(self):
        self.PYTHON_VERSION = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'

    @property
    def version(self):
        from pymor import __version__
        return __version__

    def __getattr__(self, name):
        if name.startswith('HAVE_'):
            package = name[len('HAVE_'):]
        elif name.endswith('_VERSION'):
            package = name[:-len('_VERSION')]
        else:
            raise AttributeError

        if package in _PACKAGES:
            try:
                version = _PACKAGES[package]()
            except ImportError:
                version = False

            if version is not None and version is not False:
                setattr(self, 'HAVE_' + package, True)
                setattr(self, package + '_VERSION', version)
            else:
                setattr(self, 'HAVE_' + package, False)
                setattr(self, package + '_VERSION', None)
        else:
            raise AttributeError

        return getattr(self, name)

    def __dir__(self, old=False):
        keys = set(super().__dir__())
        keys.update('HAVE_' + package for package in _PACKAGES)
        keys.update(package + '_VERSION' for package in _PACKAGES)
        return list(keys)

    def __repr__(self):
        status = {p: (lambda v: 'missing' if not v else 'present' if v is True else v)(getattr(self, p + '_VERSION'))
                  for p in _PACKAGES}
        key_width = max(len(p) for p in _PACKAGES) + 2
        package_info = [f"{p+':':{key_width}} {v}" for p, v in sorted(status.items())]
        separator = '-' * max(map(len, package_info))
        package_info = '\n'.join(package_info)
        info = f'''
pyMOR Version {self.version}

Python: {self.PYTHON_VERSION}

External Packages
{separator}
{package_info}

Defaults
--------
See pymor.core.defaults.print_defaults.
'''[1:]
        return info


config = Config()

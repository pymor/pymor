# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from importlib import import_module
from packaging.version import parse
import platform
import sys
import warnings

from pymor.core.exceptions import DependencyMissing, QtMissing, TorchMissing


def _can_import(module):
    def _can_import_single(m):
        try:
            import_module(m)
            return True
        except ImportError:
            pass
        return False
    if not isinstance(module, (list, tuple)):
        module = [module]
    return all((_can_import_single(m) for m in module))


def _get_fenics_version():
    import sys
    if "linux" in sys.platform:
        # In dolfin.__init__ the dlopen flags are set to include RTDL_GLOBAL,
        # which can cause issues with other Python C extensions.
        # In particular, with the manylinux wheels for scipy 1.9.{2,3} this leads
        # to segfaults in the Fortran L-BFGS-B implementatiton.
        #
        # A MWE to trigger the segfault is:
        #     import sys
        #     import os
        #     sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
        #     import numpy as np
        #     from scipy.optimize import minimize
        #     opt_fom_result = minimize(lambda x: x[0]**2, np.array([0.25]), method='L-BFGS-B')
        #
        # According to the comment in dolfin.__init__, setting RTLD_GLOBAL is required
        # for OpenMPI. According to the discussion in https://github.com/open-mpi/ompi/issues/3705
        # this hack is no longer necessary for OpenMPI 3.0 and later. Therefore, we save here the
        orig_dlopenflags = sys.getdlopenflags()

    import dolfin as df
    if parse(df.__version__) < parse('2019.1.0'):
        warnings.warn(f'FEniCS bindings have been tested for version 2019.1.0 and greater '
                      f'(installed: {df.__version__}).')

    if "linux" in sys.platform:
        sys.setdlopenflags(orig_dlopenflags)
    return df.__version__


def _get_dunegdt_version():
    import importlib
    version_ranges = {"dune-gdt": ('2021.1.2', '2022.2'), "dune-xt": ('2021.1.2', '2022.2')}

    def _get_version(dep_name):
        min_version, max_version = version_ranges[dep_name]
        module = importlib.import_module(dep_name.replace("-", "."))
        try:
            version = module.__version__
            if parse(version) < parse(min_version) or parse(version) >= parse(max_version):
                warnings.warn(f'{dep_name} bindings have been tested for versions between '
                              '{min_version} and {max_version} (installed: {version}).')
        except AttributeError:
            warnings.warn(f'{dep_name} bindings have been tested for versions between '
                          '{min_version} and {max_version} (installed unknown version).')
            version = None
        return version

    _get_version("dune-xt")
    return _get_version("dune-gdt")


def is_windows_platform():
    return sys.platform == 'win32' or sys.platform == 'cygwin'


def is_macos_platform():
    return 'Darwin' in platform.system()


def _get_matplotib_version():
    import matplotlib
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
        import qtpy
    except RuntimeError:
        # qtpy raises PythonQtError, which is a subclass of RuntimeError, in case no
        # Python bindings could be found. Since pyMOR always pulls qtpy, we do not want
        # to catch an ImportError
        return False
    return f'{qtpy.API_NAME} (Qt {qtpy.QT_VERSION})'


def is_jupyter():
    """Check if we believe to be running in a Jupyter Notebook or Lab.

    This method is not foolproof and might fail with any given Jupyter release.
    """
    try:
        from IPython import get_ipython
    except (ImportError, ModuleNotFoundError):
        return False
    from os import environ
    force = environ.get('PYMOR_FORCE_JUPYTER', None)
    if force is not None:
        return bool(force)
    ipy = type(get_ipython()).__module__
    return ipy.startswith('ipykernel.') or ipy.startswith('google.colab')


def is_nbconvert():
    """Check if a notebook is executed with `nbconvert`.

    In some visualization cases we need to be able to detect if a notebook
    is executed with `nbconvert` to disable async loading.
    """
    from os import environ
    return is_jupyter() and bool(environ.get('PYMOR_NBCONVERT', False))


_PACKAGES = {
    'DEALII': lambda: import_module('pymor_dealii').__version__,
    'DUNEGDT': _get_dunegdt_version,
    'FENICS': _get_fenics_version,
    'GL': lambda: import_module('OpenGL.GL') and import_module('OpenGL').__version__,
    'IPYTHON': _get_ipython_version,
    'MATPLOTLIB': _get_matplotib_version,
    'VTKIO': lambda: _can_import(('meshio', 'pyevtk', 'lxml', 'xmljson')),
    'MESHIO': lambda: import_module('meshio').__version__,
    'IPYWIDGETS': lambda: import_module('ipywidgets').__version__,
    'MPI': lambda: import_module('mpi4py.MPI') and import_module('mpi4py').__version__,
    'NGSOLVE': lambda: import_module('ngsolve').__version__,
    'NUMPY': lambda: import_module('numpy').__version__,
    'PYMESS': lambda: bool(import_module('pymess')),
    'PYTEST': lambda: import_module('pytest').__version__,
    'PYTHREEJS': lambda: import_module('pythreejs._version').__version__,
    'QT': _get_qt_version,
    'QTOPENGL': lambda: bool(_get_qt_version() and import_module('qtpy.QtOpenGL')),
    'SCIKIT_FEM': lambda: import_module('skfem').__version__,
    'SCIPY': lambda: import_module('scipy').__version__,
    'SCIPY_LSMR': lambda: hasattr(import_module('scipy.sparse.linalg'), 'lsmr'),
    'SLYCOT': lambda: _get_slycot_version(),
    'SPHINX': lambda: import_module('sphinx').__version__,
    'TORCH': lambda: import_module('torch').__version__,
    'TYPER': lambda: import_module('typer').__version__,
}


class Config:

    def __init__(self):
        self.PYTHON_VERSION = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'

    @property
    def version(self):
        from pymor import __version__
        return __version__

    def require(self, dependency):
        dependency = dependency.upper()
        if not getattr(self, f'HAVE_{dependency}'):
            if dependency == 'QT':
                raise QtMissing
            elif dependency == 'TORCH':
                raise TorchMissing
            else:
                raise DependencyMissing(dependency)

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

Python {self.PYTHON_VERSION} on {platform.platform()}

External Packages
{separator}
{package_info}

Defaults
--------
See pymor.core.defaults.print_defaults.
'''[1:]
        return info


config = Config()

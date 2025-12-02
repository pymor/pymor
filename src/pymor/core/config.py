# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
import platform
import sys
import warnings
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse

from pymor.core.exceptions import DependencyMissingError, QtMissingError, TorchMissingError


def _can_import(module):
    def _can_import_single(m):
        try:
            import_module(m)
            return True
        except ImportError:
            pass
        return False
    if not isinstance(module, list | tuple):
        module = [module]
    return all(_can_import_single(m) for m in module)


def _get_fenics_version():
    import sys
    if 'linux' in sys.platform:
        # In dolfin.__init__ the dlopen flags are set to include RTDL_GLOBAL,
        # which can cause issues with other Python C extensions.
        # In particular, with the manylinux wheels for scipy 1.9.{2,3} this leads
        # to segfaults in the Fortran L-BFGS-B implementation.
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

    if 'linux' in sys.platform:
        sys.setdlopenflags(orig_dlopenflags)
    return df.__version__


def _get_fenicsx_version():
    import dolfinx as dfx
    if not (parse('0.10') <= parse(dfx.__version__) < parse('0.11')):
        warnings.warn(f'FEniCSx bindings have only been tested for version 0.10 '
                      f'(installed: {dfx.__version__}).')
    return dfx.__version__

def is_windows_platform():
    return sys.platform == 'win32' or sys.platform == 'cygwin'


def is_macos_platform():
    return 'Darwin' in platform.system()


def is_scipy_mkl():
    return 'mkl' in config.SCIPY_INFO


def _get_threadpool_internal_api(module):
    from subprocess import run
    result = run(
        [sys.executable, '-c', f'from threadpoolctl import threadpool_info as tpi; import {module};\n'
                                'for d in tpi(): print(d["internal_api"])'],
        capture_output=True
    )
    return {x.strip() for x in result.stdout.decode().split('\n') if x.strip()}


def _get_version(module, threadpoolctl_internal_api=False):
    def impl():
        version = import_module(module).__version__
        if threadpoolctl_internal_api:
            try:
                info = ', '.join(_get_threadpool_internal_api(module))
                return version, info
            except OSError:
                return version
        else:
            return version
    return impl


def _get_matplotlib_version():
    import matplotlib as mpl
    return mpl.__version__


def _get_slycot_version():
    from slycot.version import version
    if list(map(int, version.split('.'))) < [0, 3, 1]:
        import warnings
        warnings.warn('Slycot support disabled (version 0.3.1 or higher required).')
        return False
    else:
        info = ', '.join(_get_threadpool_internal_api('slycot'))
        return version, info


def _get_qt_version():
    try:
        import qtpy
    except RuntimeError:
        # qtpy raises PythonQtError, which is a subclass of RuntimeError, in case no
        # Python bindings could be found. Since pyMOR always pulls qtpy, we do not want
        # to catch an ImportError
        return False
    return f'{qtpy.API_NAME} (Qt {qtpy.QT_VERSION})'


def _get_umfpack_version():
    try:
        return version('scikit-umfpack')
    except PackageNotFoundError:
        pass
    return False


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
    return ipy.startswith('ipykernel.') or ipy.startswith('google.colab') or ipy.startswith('pyolite.')


_PACKAGES = {
    'DEALII': _get_version('pymor_dealii'),
    'FENICS': _get_fenics_version,
    'FENICSX': _get_fenicsx_version,
    'GL': lambda: import_module('OpenGL.GL') and import_module('OpenGL').__version__,
    'IPYPARALLEL': _get_version('ipyparallel'),
    'IPYTHON': _get_version('IPython'),
    'IPYWIDGETS': _get_version('ipywidgets'),
    'K3D': _get_version('k3d'),
    'MATPLOTLIB': _get_matplotlib_version,
    'MESHIO': _get_version('meshio'),
    'MPI': lambda: import_module('mpi4py.MPI') and import_module('mpi4py').__version__,
    'NGSOLVE': _get_version('ngsolve', True),
    'NUMPY': _get_version('numpy', True),
    'PYTEST': _get_version('pytest'),
    'QT': _get_qt_version,
    'QTOPENGL': lambda: bool(_get_qt_version() and import_module('qtpy.QtOpenGL')),
    'SCIKIT_FEM': _get_version('skfem'),
    'SCIPY': _get_version('scipy', True),
    'SLYCOT': _get_slycot_version,
    'SPHINX': _get_version('sphinx'),
    'TORCH': _get_version('torch', True),
    'THREADPOOLCTL': _get_version('threadpoolctl'),
    'TYPER': _get_version('typer'),
    'UMFPACK': _get_umfpack_version,
    'VTKIO': lambda: _can_import(('meshio', 'pyevtk', 'lxml', 'xmljson')),
}


class Config:

    def __init__(self):
        self.PYTHON_VERSION = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
        self.disabled = set(os.environ.get('PYMOR_CONFIG_DISABLE', '').upper().split())

    @property
    def version(self):
        from pymor import __version__
        return __version__

    def require(self, dependency):
        dependency = dependency.upper()
        if not getattr(self, f'HAVE_{dependency}'):
            if dependency == 'QT':
                raise QtMissingError
            elif dependency == 'TORCH':
                raise TorchMissingError
            else:
                raise DependencyMissingError(dependency)

    def __getattr__(self, name):
        if name.startswith('HAVE_'):
            package = name[len('HAVE_'):]
        elif name.endswith('_VERSION'):
            package = name[:-len('_VERSION')]
        elif name.endswith('_INFO'):
            package = name[:-len('_INFO')]
        else:
            raise AttributeError

        if package in _PACKAGES:
            if package in self.disabled:
                version = None
                info = None
                status = 'disabled'
            else:
                try:
                    result = _PACKAGES[package]()
                    if isinstance(result, tuple):
                        assert len(result) == 2
                        version, info = result
                    else:
                        version, info = result, None
                    if not version:
                        raise ImportError
                    status = 'present'
                except ImportError:
                    version = None
                    info = None
                    status = 'missing'
                except Exception:
                    version = None
                    info = None
                    status = 'import check failed'

            setattr(self, 'HAVE_' + package, version is not None)
            setattr(self, package + '_VERSION', version)
            setattr(self, package + '_INFO', info)
            setattr(self, package + '_STATUS', status)
        else:
            raise AttributeError

        return getattr(self, name)

    def __dir__(self):
        keys = set(super().__dir__())
        keys.update('HAVE_' + package for package in _PACKAGES)
        keys.update(package + '_VERSION' for package in _PACKAGES)
        keys.update(package + '_INFO' for package in _PACKAGES)
        return list(keys)

    def __repr__(self):

        def get_status(p):
            version = getattr(self, p + '_VERSION')
            if not version or version is True:
                result = getattr(self, p + '_STATUS')
            else:
                result = version
            info = getattr(self, p + '_INFO')
            if info:
                return f'{result} ({info})'
            else:
                return result

        status = {p: get_status(p) for p in _PACKAGES}
        key_width = max(len(p) for p in _PACKAGES) + 2
        package_info = [f"{p+':':{key_width}} {v}" for p, v in sorted(status.items())]
        separator = '-' * max(map(len, package_info))
        package_info = '\n'.join(package_info)
        info = f"""
pyMOR Version {self.version}

Python {self.PYTHON_VERSION} on {platform.platform()}

External Packages
{separator}
{package_info}

Defaults
--------
See pymor.core.defaults.print_defaults.

Caching
-------
See pymor.core.cache.print_cached_methods.
"""[1:]
        return info


config = Config()

#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

# DO NOT use any python features here that require 3.6 or newer

_PYTEST = 'pytest==7.1.2'
# 5.12.* blocked due to https://bugreports.qt.io/browse/PYSIDE-1004
# however the problem is not actually fixed in 5.12.3 as advertised,
# but only starting from 5.13.1
_PYSIDE = 'PySide2>=5.15.2.1'


def _numpy_scipy():
    # numpy versions with filters according to minimal version with a wheel
    # 1.24 limit due to https://github.com/pymor/pymor/issues/1692
    numpys = [
        'numpy>=1.17.5,<1.24;python_version == "3.8"',
        'numpy>=1.19.4,<1.24;python_version >= "3.9"',
    ]
    scipys = [
        'scipy>=1.3;python_version < "3.8"',
        'scipy>=1.3.3;python_version == "3.8"',
        'scipy>=1.5.4;python_version >= "3.9"',
    ]
    return numpys + scipys


def setup_requires():
    return [
        'setuptools',
        'wheel',
        'pytest-runner>=2.9',
        'packaging',
    ]


# recheck if jupyter_client pin still necessary
#   https://github.com/jupyter-widgets/pythreejs/issues/366
# Qt bindings selectors are a woraround for https://bugreports.qt.io/browse/QTBUG-88688
install_requires = ['qtpy!=2.0.0', 'packaging', 'diskcache', 'typer', 'click'] + _numpy_scipy()
install_suggests = {
    'ipython>=5.0': 'an enhanced interactive python shell',
    'ipyparallel>=6.2.5': 'required for pymor.parallel.ipython',
    'matplotlib': 'needed for error plots in demo scipts',
    'pyopengl': 'fast solution visualization for builtin discretizations (PySide also required)',
    'sympy': 'symbolic mathematics',
    'pygments': 'highlighting code',
    'pythreejs': 'threejs bindings for python notebook  visualization',
    'jupyter_client>=7.0.6': 'necessary to explicitly state here to fix 3js',
    _PYTEST: 'testing framework required to execute unit tests',
    _PYSIDE: 'solution visualization for builtin discretizations',
    'ipywidgets': 'notebook GUI elements',
    'nbresuse': 'resource usage indicator for notebooks',
    'torch': 'PyTorch open source machine learning framework',
    'jupyter_contrib_nbextensions': 'modular collection of jupyter extensions',
    'pillow': 'image library used for bitmap data functions',
    'dune-gdt>=2021.1.3; platform_system=="Linux" and platform_machine=="x86_64"': 'generic discretization toolbox',
    'dune-xt>=2021.1.3; platform_system=="Linux" and platform_machine=="x86_64"': 'DUNE extensions for dune-gdt',
}
io_requires = ['pyevtk', 'xmljson', 'meshio>=4.4', 'lxml', 'gmsh']
install_suggests.update({p: 'optional File I/O support libraries' for p in io_requires})
doc_requires = ['sphinx>=5.0', 'matplotlib', _PYSIDE, 'ipyparallel>=6.2.5', 'python-slugify',
                'ipywidgets', 'sphinx-qt-documentation', 'bash_kernel', 'sphinx-material',
                'sphinxcontrib-bibtex', 'sphinx-autoapi>=1.8', 'myst-nb>=0.16'] + install_requires
ci_requires = ['check-manifest==0.48',
               'check_reqs==0.2.0',
               'codecov==2.1.12',
               'docutils==0.18.1',
               'flake8-docstrings==1.6.0',
               'flake8-rst-docstrings==0.2.6',
               'hypothesis[numpy,pytest]==6.48.2',
               'pybind11==2.9.2',
               'pypi-oldest-requirements==2021.2.0',
               'pyqt5-qt5==5.15.2',
               'pyqt5==5.15.7',
               _PYTEST,
               'pytest-cov==3.0.0',
               'pytest-memprof==0.2.0',
               'pytest-parallel==0.1.1',
               'pytest-regressions==2.3.1',
               'pytest-xdist==2.5.0',
               'readme_renderer[md]==35.0',
               'rstcheck==6.0.0.post1',
               'scikit-fem==6.0.0',
               'twine==3.8.0']

# Slycot is pinned due to buildsystem changes + missing wheels
optional_requirements_file_only = (['slycot>=0.4.0', 'pymess',
                                    'mpi4py>=3.0.3;python_version >= "3.9"',
                                    'mpi4py>=3.0;python_version < "3.9"'])


def strip_markers(name):
    for m in ';<>=':
        try:
            i = name.index(m)
            name = name[:i].strip()
        except ValueError:
            continue
    return name


def extras():
    import pkg_resources
    import itertools

    def _candidates(blocklist):
        # skip those which aren't needed in our current environment (py ver, platform)
        for pkg in set(itertools.chain(doc_requires, install_suggests.keys())):
            if pkg in blocklist:
                continue
            try:
                marker = next(pkg_resources.parse_requirements(pkg)).marker
                if marker is None or marker.evaluate():
                    yield pkg
            except pkg_resources.RequirementParseError:
                # try to fake a package to get the marker parsed
                stripped = strip_markers(pkg)
                fake_pkg = 'pip ' + pkg.replace(stripped, '')
                try:
                    marker = next(pkg_resources.parse_requirements(fake_pkg)).marker
                    if marker is None or marker.evaluate():
                        yield pkg
                except pkg_resources.RequirementParseError:
                    continue

    # blocklisted packages need a (minimal) compiler setup
    # - nbresuse, pytest-memprof depend on psutil which has no wheels
    # - slycot directly needs a compiler setup with BLAS, plus scikit-build + cmake
    # - pymess is better installed from source (see README.md)
    return {
        'full': list(_candidates(blocklist=['slycot', 'pymess', 'nbresuse', 'pytest-memprof'])),
        'ci':  ci_requires,
        'docs': doc_requires,
        'io': io_requires,
    }


toml_tpl = '''
[build-system]
requires = {0}
build-backend = "setuptools.build_meta"
'''
if __name__ == '__main__':
    note = '# This file is autogenerated. Edit dependencies.py instead'
    import os
    import itertools
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'wt') as req:
        req.write(note+'\n')
        for module in sorted(set(itertools.chain(install_requires, setup_requires()))):
            req.write(module+'\n')
    with open(os.path.join(os.path.dirname(__file__), 'requirements-optional.txt'), 'wt') as req:
        req.write(note+'\n')
        req.write('-r requirements.txt\n')
        req.write('-r requirements-ci.txt\n')
        for module in sorted(set(itertools.chain(optional_requirements_file_only,
                                                 doc_requires, install_suggests.keys()))):
            req.write(module+'\n')
    with open(os.path.join(os.path.dirname(__file__), 'requirements-ci.txt'), 'wt') as req:
        req.write('-r requirements.txt\n')
        req.write(note+'\n')
        for module in sorted(ci_requires):
            req.write(module+'\n')
    with open(os.path.join(os.path.dirname(__file__), 'pyproject.toml'), 'wt') as toml:
        toml.write(note)
        toml.write(toml_tpl.format(str(setup_requires())))

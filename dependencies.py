#!/usr/bin/env python3
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# DO NOT use any python features here that require 3.6 or newer

_PYTEST = 'pytest>=4.4'

def _pymess(rev, major, minor, marker=True):
    url = 'https://www.mpi-magdeburg.mpg.de/mpcsc/software/cmess/{rev}/pymess-{rev}-cp{major}{minor}-cp{major}{minor}m-manylinux1_x86_64.whl'
    # tmp workaround till next release
    url = 'https://pymor.github.io/wheels/pymess-{rev}-cp{major}{minor}-cp{major}{minor}m-manylinux1_x86_64.whl'
    url = url.format(rev=rev, major=major, minor=minor)
    if marker:
        return '{url} ; python_version == "{major}.{minor}" and "linux" in sys_platform'.format(url=url, major=major, minor=minor)
    return url

# for pyproject.toml we require equality to build compatible wheels in pep 517 mode
def setup_requires(toml=False):
    NUMPY = '1.16.0'
    # numpy versions with filters according to minimal version with a wheel
    numpys = ['numpy>={};python_version == "3.6"'.format(NUMPY),
      'numpy>=1.15.4;python_version == "3.7"',
      'numpy>=1.17.5;python_version == "3.8"',
      'numpy>={};python_version != "3.6" and python_version != "3.7" and python_version != "3.8"'.format(NUMPY),]
    other = ['setuptools>=40.8.0', 'wheel', 'pytest-runner>=2.9', 'cython>=0.27', 'packaging',]
    if toml:
        numpys = [f.replace('numpy>=', 'numpy==') for f in numpys]
    return numpys + other

install_requires = ['scipy>=1.1', 'Qt.py>=1.2.4', 'packaging','diskcache', 'docopt-ng'] + setup_requires()
install_suggests = {'ipython>=5.0': 'an enhanced interactive python shell',
                    'ipyparallel': 'required for pymor.parallel.ipython',
                    'matplotlib': 'needed for error plots in demo scipts',
                    'meshio==3.3.1': 'needed to import Gmsh grids',
                    'pyopengl': 'fast solution visualization for builtin discretizations (PySide also required)',
                    'pyamg': 'algebraic multigrid solvers',
                    'pyevtk>=1.1': 'writing vtk output',
                    'pygmsh==6.0.2': 'python frontend for gmsh',
                    'sympy': 'symbolic mathematics',
                    'pythreejs': 'threejs bindings for python notebook  visualization',
                    _PYTEST: 'testing framework required to execute unit tests',
                    'PySide2': 'solution visualization for builtin discretizations',
                    'ipywidgets': 'notebook GUI elements',
                    'pillow': 'image library used for bitmap data functions'}
doc_requires = ['sphinx>=1.7', 'pymor-nb2plots>=0.7', 'matplotlib', 'PySide2', 'ipyparallel',
                'ipywidgets', 'sphinx-qt-documentation'] + install_requires
ci_requires = [_PYTEST, 'pytest-cov', 'pytest-xdist', 'check-manifest', 'nbconvert', 'pytest-parallel',
               'readme_renderer[md]', 'rstcheck', 'codecov', 'twine', 'pytest-memprof', 'pytest-timeout', 
               'testipynb', "pypi-oldest-requirements>=2020.2", 'pytest-deadfixtures']
import_names = {'ipython': 'IPython',
                'pytest-cache': 'pytest_cache',
                'pytest-instafail': 'pytest_instafail',
                'pytest-xdist': 'xdist',
                'pytest-cov': 'pytest_cov',
                'pytest-flakes': 'pytest_flakes',
                'pytest-pep8': 'pytest_pep8',
                _pymess('1.0.0', 3, 6, False): 'pymess',
                _pymess('1.0.0', 3, 7, False): 'pymess',
                'pyopengl': 'OpenGL'}
# Slycot is pinned due to buildsystem changes + missing wheels
optional_requirements_file_only = [_pymess('1.0.0', 3, 6),_pymess('1.0.0', 3, 7),
                    'slycot==0.3.3', 'mpi4py']

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

    def _candidates(blacklist):
        # skip those which aren't needed in our current environment (py ver, platform)
        for pkg in set(itertools.chain(doc_requires, install_suggests.keys())):
            if pkg in blacklist:
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

    return {
        'full': list(_candidates(blacklist=[])),
        'ci':  ci_requires,
        'docs': doc_requires,
    }

toml_tpl = '''
[build-system]
requires = {0}
build-backend = "setuptools.build_meta"
'''
if __name__ == '__main__':
    note = '# This file is autogenerated. Edit dependencies.py instead'
    print(' '.join([i for i in install_requires + list(install_suggests.keys())]))
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
        toml.write(toml_tpl.format(str(setup_requires(toml=True))))

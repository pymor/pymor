#!/usr/bin/env python3
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

_PYTEST = 'pytest>=3.3'

def _pymess(rev, major, minor, marker=True):
    try:
        # direct urls are only supported with newer pip and when not installing pymor from pypi
        import pip
        from distutils.version import StrictVersion
        pref = 'pymess@' if StrictVersion(pip.__version__) >= StrictVersion('18.1') else ''
    except:
        pref = ''
    url = '{pref}https://www.mpi-magdeburg.mpg.de/mpcsc/software/cmess/{rev}/pymess-{rev}-cp{major}{minor}-cp{major}{minor}m-manylinux1_x86_64.whl'
    # tmp workaround till next release
    url = '{pref}https://pymor.github.io/wheels/pymess-{rev}-cp{major}{minor}-cp{major}{minor}m-manylinux1_x86_64.whl'
    url = url.format(rev=rev, major=major, minor=minor, pref=pref)
    if marker:
        return '{url} ; python_version == "{major}.{minor}" and "linux" in sys_platform'.format(url=url, major=major, minor=minor)
    return url

tests_require = [_PYTEST, 'pytest-cov', 'envparse', 'docker']
install_requires = ['cython>=0.20.1', 'numpy>=1.8.1', 'scipy>=0.13.3', 'Sphinx>=1.4.0', 'docopt', 'Qt.py', 'packaging']
setup_requires = ['pytest-runner>=2.9', 'cython>=0.20.1', 'numpy>=1.8.1', 'packaging']
install_suggests = {'ipython>=3.0': 'an enhanced interactive python shell',
                    'ipyparallel': 'required for pymor.parallel.ipython',
                    'matplotlib': 'needed for error plots in demo scipts',
                    'pyopengl': 'fast solution visualization for builtin discretizations (PySide also required)',
                    'pyamg': 'algebraic multigrid solvers',
                    'mpi4py': 'required for pymor.tools.mpi and pymor.parallel.mpi',
                    'pyevtk>=1.1': 'writing vtk output',
                    _PYTEST: 'testing framework required to execute unit tests',
                    _pymess('1.0.0', 3, 5): 'Python bindings for M.E.S.S. (Matrix Equation Sparse Solver)',
                    _pymess('1.0.0', 3, 6): 'Python bindings for M.E.S.S. (Matrix Equation Sparse Solver)',
                    _pymess('1.0.0', 3, 7): 'Python bindings for M.E.S.S. (Matrix Equation Sparse Solver)',
                    'PyQt5': 'solution visualization for builtin discretizations',
                    'pillow': 'image library used for bitmap data functions',
                    'psutil': 'Process management abstractions used for gui',
                    'slycot>=0.3.3': 'python wrapper for the SLICOT control and systems library'}
doc_requires = ['sphinx>=1.5', 'cython', 'numpy']
travis_requires = ['pytest-cov', 'pytest-xdist', 'check-manifest', 'codecov', 'pytest-travis-fold']
import_names = {'ipython': 'IPython',
                'pytest-cache': 'pytest_cache',
                'pytest-instafail': 'pytest_instafail',
                'pytest-xdist': 'xdist',
                'pytest-cov': 'pytest_cov',
                'pytest-flakes': 'pytest_flakes',
                'pytest-pep8': 'pytest_pep8',
                _pymess('1.0.0', 3, 5, False): 'pymess',
                _pymess('1.0.0', 3, 6, False): 'pymess',
                _pymess('1.0.0', 3, 7, False): 'pymess',
                'pyopengl': 'OpenGL'}
needs_extra_compile_setup = ['mpi4py']


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

    def _ex(name):
        # no environment specifiers or wheel URI etc are allowed in extras
        name = strip_markers(name)
        try:
            next(pkg_resources.parse_requirements(name))
        except pkg_resources.RequirementParseError:
            name = import_names[name]
        return name

    def _candidates(blacklist):
        # skip those which aren't needed in our current environment (py ver, platform)
        for pkg in set(itertools.chain(doc_requires, tests_require, install_suggests.keys())):
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
        'full-nompi': [_ex(f) for f in _candidates(blacklist=needs_extra_compile_setup)],
        'full': [_ex(f) for f in _candidates(blacklist=[])],
        'travis':  travis_requires,
    }


if __name__ == '__main__':
    note = '# This file is autogenerated. Edit dependencies.py instead'
    print(' '.join([i for i in install_requires + list(install_suggests.keys())]))
    import os
    import itertools
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'wt') as req:
        req.write(note+'\n')
        for module in sorted(set(itertools.chain(install_requires, setup_requires))):
            req.write(module+'\n')
    with open(os.path.join(os.path.dirname(__file__), 'requirements-optional.txt'), 'wt') as req:
        req.write(note+'\n')
        req.write('-r requirements.txt\n')
        for module in sorted(set(itertools.chain(tests_require, install_suggests.keys()))):
            req.write(module+'\n')
    with open(os.path.join(os.path.dirname(__file__), 'requirements-rtd.txt'), 'wt') as req:
        rtd = '''# This file is sourced by readthedocs.org to install missing dependencies.
# We need a more recent version of Sphinx for being able to provide
# our own docutils.conf.'''
        req.write(rtd+'\n')
        req.write(note+'\n')
        for module in sorted(doc_requires):
            req.write(module+'\n')
    with open(os.path.join(os.path.dirname(__file__), 'requirements-travis.txt'), 'wt') as req:
        req.write('-r requirements.txt\n')
        req.write(note+'\n')
        for module in sorted(travis_requires):
            req.write(module+'\n')

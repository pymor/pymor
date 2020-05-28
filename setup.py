#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# DO NOT use any python features here that require 3.6 or newer

import sys
import os

from setuptools import find_packages
from distutils.extension import Extension
from distutils.command.build_py import build_py as _build_py
import itertools
# versioneer does not work in a pep518/7 context w/o modification here
sys.path.append(os.path.dirname(__file__))
import versioneer
import pprint

import dependencies

tests_require = dependencies.ci_requires
install_requires = dependencies.install_requires
setup_requires = dependencies.setup_requires()
install_suggests = dependencies.install_suggests


class DependencyMissing(Exception):

    def __init__(self, names):
        super().__init__('Try: "for i in {} ; do pip install $i ; done"'.format(' '.join(names)))


def _numpy_monkey():
    '''Apparently we need to monkey numpy's distutils to be able to build
    .pyx with Cython instead of Pyrex. The monkeying below is copied from
    https://github.com/matthew-brett/du-cy-numpy/blob/master/matthew_monkey.py
    via the discussion at http://comments.gmane.org/gmane.comp.python.numeric.general/37752
    '''
    global _build_src
    from os.path import join as pjoin, dirname
    from distutils.dep_util import newer_group
    from distutils.errors import DistutilsError

    from numpy.distutils.misc_util import appendpath
    from numpy.distutils import log

    from numpy.distutils.command import build_src
    _orig_generate_a_pyrex_source = build_src.build_src.generate_a_pyrex_source

    def generate_a_pyrex_source(self, base, ext_name, source, extension):
        ''' Monkey patch for numpy build_src.build_src method

        Uses Cython instead of Pyrex, iff source contains 'pymor'

        Assumes Cython is present
        '''
        if 'pymor' not in source:
            return _orig_generate_a_pyrex_source(self, base, ext_name, source, extension)

        if self.inplace:
            target_dir = dirname(base)
        else:
            target_dir = appendpath(self.build_src, dirname(base))
        target_file = pjoin(target_dir, ext_name + '.c')
        depends = [source] + extension.depends
        if self.force or newer_group(depends, target_file, 'newer'):
            import Cython.Compiler.Main
            log.info("cythonc:> %s" % (target_file))
            self.mkpath(target_dir)
            options = Cython.Compiler.Main.CompilationOptions(
                defaults=Cython.Compiler.Main.default_options,
                include_path=extension.include_dirs,
                output_file=target_file)
            cython_result = Cython.Compiler.Main.compile(source, options=options)
            if cython_result.num_errors != 0:
                raise DistutilsError("%d errors while compiling %r with Cython"
                                     % (cython_result.num_errors, source))
        return target_file

    build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source



cmdclass = versioneer.get_cmdclass()



def _testdatafiles():
    root = os.path.join(os.path.dirname(__file__), 'src', 'pymortests')
    testdata = set()

    for dir_, _, files in os.walk(os.path.join(root, 'testdata')):
        for fileName in files:
            relDir = os.path.relpath(dir_, root)
            relFile = os.path.join(relDir, fileName)
            testdata.add(relFile)
    return list(testdata)


def _setup(**kwargs):
    # the following hack is taken from scipy's setup.py
    # https://github.com/scipy/scipy
    if (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1] in ('--help-commands', 'egg_info', '--version', 'clean'))):
        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Scipy when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
        return setup(**kwargs)

    _numpy_monkey()
    import Cython.Distutils
    # numpy sometimes expects this attribute, sometimes not. all seems ok if it's set to none
    if not hasattr(Cython.Distutils.build_ext, 'fcompiler'):
        Cython.Distutils.build_ext.fcompiler = None
    cmdclass.update({'build_ext': Cython.Distutils.build_ext})
    # setuptools sdist command my to include some files apparently (https://github.com/numpy/numpy/pull/7131)
    from distutils.command.sdist import sdist
    cmdclass.update({'sdist': sdist})
    from numpy import get_include
    include_dirs = [get_include()]
    ext_modules = [Extension("pymor.discretizers.builtin.relations", ["src/pymor/discretizers/builtin/relations.pyx"], include_dirs=include_dirs),
                   Extension("pymor.discretizers.builtin.inplace", ["src/pymor/discretizers/builtin/inplace.pyx"], include_dirs=include_dirs),
                   Extension("pymor.discretizers.builtin.grids._unstructured", ["src/pymor/discretizers/builtin/grids/_unstructured.pyx"], include_dirs=include_dirs)]
    # for some reason the *pyx files don't end up in sdist tarballs -> manually add them as package data
    # this _still_ doesn't make them end up in the tarball however -> manually add them in MANIFEST.in
    # duplication is necessary since Manifest sometime is only regarded in sdist, package_data in bdist
    # all filenames need to be relative to their package root, not the source root
    kwargs['package_data'] = {'pymor': [f.replace('src/pymor/', '') for f in itertools.chain(*[i.sources for i in ext_modules])],
                              'pymortests': _testdatafiles()}

    kwargs['cmdclass'] = cmdclass
    kwargs['ext_modules'] = ext_modules

    # lastly we'll need to tweak matplotlibs config dir or else
    # installing it from setup will result in a SandboxViolation
    import os
    os.environ['MPLCONFIGDIR'] = "."

    from numpy.distutils.core import setup
    return setup(**kwargs)


def setup_package():

    _setup(
        name='pymor',
        version=versioneer.get_version(),
        author='pyMOR developers',
        author_email='pymor-dev@listserv.uni-muenster.de',
        maintainer='Rene Fritze',
        maintainer_email='rene.fritze@wwu.de',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        include_package_data=True,
        scripts=['src/pymor-demo', 'dependencies.py'],
        url='http://pymor.org',
        description=' ',
        python_requires='>=3.6',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        tests_require=tests_require,
        install_requires=install_requires,
        extras_require = dependencies.extras(),
        classifiers=['Development Status :: 4 - Beta',
                     'License :: OSI Approved :: BSD License',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Intended Audience :: Science/Research',
                     'Topic :: Scientific/Engineering :: Mathematics'],
        license='LICENSE.txt',
        zip_safe=False,
        cmdclass=cmdclass,
    )


if __name__ == '__main__':
    setup_package()

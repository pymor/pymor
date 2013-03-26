#!/usr/bin/env python

#make sure we got distribute in place
from distribute_setup import use_setuptools
use_setuptools()

import sys
from setuptools import setup
from distutils.extension import Extension

tests_requires = ['mock', 'nose-cov', 'nose', 'nosehtmloutput', 'nose-progressive', 'tissue']
install_requires = ['distribute', 'scipy', 'matplotlib', 'numpy', 'pycontracts', 'sympy', 'docopt' ] + tests_requires

cmdclass = {}
ext_modules = []
try:
    import Cython.Distutils
    cmdclass['build_ext'] = Cython.Distutils.build_ext
    ext_modules.append(Extension("pymor.tools.relations", ["src/pymor/tools/relations.pyx"]))
except ImportError:
    sys.stderr.write('*' * 79 + '\n')
    sys.stderr.write('Failed to import build_ext from cython, no extension modules will be build.\n')
    sys.stderr.write('*' * 79 + '\n')
    
setup(
    name = 'pyMor',
    version = '0.0.1',
    author = 'pyMor developers',
    author_email = 'pymor-dev@listserv.uni-muenster.de',
    maintainer = 'Rene Milk',
    maintainer_email = 'rene.milk@wwu.de',
    package_dir = {'': 'src'},
    packages = ['pymor', 'pymortests'],
    scripts = ['bin/%s'%n for n in [] ] + ['run_tests.py'],
    url = 'http://dune-project.uni-muenster.de/git/projects/pymor',
    description = ' ',
    long_description = open('README.txt').read(),
    tests_require = tests_requires,
    install_requires = install_requires,
    classifiers = ['Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization'],
    license = 'LICENSE.txt',
    #cmdclass = cmdclass,
    ext_modules = ext_modules,
)

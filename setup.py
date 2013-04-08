#!/usr/bin/env python
# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#make sure we got distribute in place
from distribute_setup import use_setuptools
use_setuptools()

import sys
import os
import subprocess
from setuptools import setup, find_packages
from distutils.extension import Extension

def write_version():
    revstring = subprocess.check_output(['git', 'describe', '--tags', '--candidates', '20', '--match', '*.*.*']).strip()
    filename = os.path.join(os.path.dirname(__file__), 'src', 'pymor', 'version.py')
    with open(filename, 'w') as out:
        out.write('revstring = \'{}\''.format(revstring))

def setup_package():
    write_version()

    tests_requires = ['mock', 'nose-cov', 'nose', 'nosehtmloutput', 'nose-progressive', 'tissue>=0.8']
    install_requires = ['distribute', 'scipy', 'matplotlib', 'numpy', 'pycontracts',
                        'sympy', 'docopt', 'dogpile.cache' ] + tests_requires
    
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
        name='pyMor',
        version='0.1.0',
        author='pyMor developers',
        author_email='pymor-dev@listserv.uni-muenster.de',
        maintainer='Rene Milk',
        maintainer_email='rene.milk@wwu.de',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        scripts=['bin/%s' % n for n in [] ] + ['run_tests.py'],
        url='http://pymor.org',
        description=' ' ,
        long_description=open('README.txt').read(),
        tests_require=tests_requires,
        install_requires=install_requires,
        classifiers=['Development Status :: 2 - Pre-Alpha',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Visualization'],
        license = 'LICENSE.txt',
        cmdclass=cmdclass,
        ext_modules=ext_modules,
    )

if __name__ == '__main__':
    setup_package()

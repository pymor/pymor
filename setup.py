#!/usr/bin/env python
import sys
from distutils.core import setup

tests_require = ['mock', 'nose-cov', 'nose', 'nosehtmloutput', 'nose-progressive']

setup(
		name = 'pyMor',
		version = '0.0.1',
		author = 'pyMor developers',
		author_email = 'pymor-dev@listserv.uni-muenster.de',
		maintainer = 'Rene Milk',
		maintainer_email = 'rene.milk@wwu.de',
		package_dir = {'': 'src'},
		packages = ['pymor'],
		scripts = ['bin/%s'%n for n in [] ] + ['run_tests.py'],
		url = 'http://dune-project.uni-muenster.de/git/projects/pymor',
		description = ' ',
		long_description = open('README.txt').read(),
		tests_require = tests_require,
		# running `setup.py sdist' gives a warning about this, but
		# install_requires is the only thing that works with pip/easy_install...
		# we do not list pyqt here since pip can't seem to install it
		install_requires = ['scipy', 'matplotlib', 'numpy', 'pycontracts', 'sympy' ] + tests_require,
		classifiers = ['Development Status :: 2 - Pre-Alpha',
				'License :: OSI Approved :: BSD License',
				'Programming Language :: Python :: 2.7',
				'Programming Language :: Python :: 3',
				'Intended Audience :: Science/Research',
				'Topic :: Scientific/Engineering :: Mathematics',
				'Topic :: Scientific/Engineering :: Visualization'],
		license = 'LICENSE.txt',
)

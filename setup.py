import sys
from distutils.core import setup

tests_require = ['nose-cov', 'nose', 'nosehtmloutput' ]

setup(
    name = 'pyMor',
    version = '0.0.1',
    author = 'Rene Milk',
    author_email = 'rene.milk@uni-muenster.de',
    packages = ['pymor'],
    scripts = ['bin/%s'%n for n in [] ],
    url = 'http://dune-project.uni-muenster.de/git/projects/pymor',
    description = ' ',
    long_description = open('README.txt').read(),
    tests_require = tests_require,
    # running `setup.py sdist' gives a warning about this, but 
    # install_requires is the only thing that works with pip/easy_install...
    # we do not list pyqt here since pip can't seem to install it
    install_requires = ['matplotlib', 'numpy' ] + tests_require,
    classifiers = ['Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization'],
    license = 'LICENSE.txt'
)

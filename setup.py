#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

# DO NOT use any python features here that require 3.6 or newer

import sys
import os
from setuptools import setup, find_packages
# versioneer (+dependencies) does not work in a pep518/7 context w/o modification here
sys.path.insert(0, os.path.dirname(__file__))
import versioneer  # noqa
import dependencies  # noqa

tests_require = dependencies.ci_requires
install_requires = dependencies.install_requires
setup_requires = dependencies.setup_requires()
install_suggests = dependencies.install_suggests
# we'll need to tweak matplotlibs config dir or else
# installing it from setup will result in a SandboxViolation
os.environ['MPLCONFIGDIR'] = "."


class DependencyMissing(Exception):

    def __init__(self, names):
        super().__init__('Try: "for i in {} ; do pip install $i ; done"'.format(' '.join(names)))


def _testdatafiles():
    root = os.path.join(os.path.dirname(__file__), 'src', 'pymortests')
    testdata = set()

    for dir_, _, files in os.walk(os.path.join(root, 'testdata')):
        for fileName in files:
            relDir = os.path.relpath(dir_, root)
            relFile = os.path.join(relDir, fileName)
            testdata.add(relFile)
    return list(testdata)


def setup_package():
    # DO NOT modify the sdist class, it breaks versioneer
    # the warning about using distutils.command.sdist is a lie
    # versioneer already wraps it internally
    # all filenames need to be relative to their package root, not the source root

    setup(
        name='pymor',
        version=versioneer.get_version(),
        author='pyMOR developers',
        author_email='main.developers@pymor.org',
        maintainer='Rene Fritze',
        maintainer_email='rene.fritze@wwu.de',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'pymor-demo = pymor.scripts.pymor_demo:run',
                'pymor-vis = pymor.scripts.pymor_vis:run',
            ],
        },
        url='https://pymor.org',
        description=' ',
        python_requires='>=3.8',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        tests_require=tests_require,
        install_requires=install_requires,
        extras_require=dependencies.extras(),
        classifiers=['Development Status :: 4 - Beta',
                     'License :: OSI Approved :: BSD License',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Intended Audience :: Science/Research',
                     'Topic :: Scientific/Engineering :: Mathematics'],
        license='LICENSE.txt',
        zip_safe=False,
        cmdclass=versioneer.get_cmdclass(),
        package_data={
            'pymortests': _testdatafiles(),
        },
    )


if __name__ == '__main__':
    setup_package()

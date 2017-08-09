#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Debian package builder

Usage:
  all.py [-burh] [-v REV] TAG

Arguments:
  TAG                       Commit/Branch/Tag to checkout

Options:
  -b, --binary              Also build binary packages for host arch. Requires
                            docker and debocker installed and usable.
  -u, --upload              Upload source package to Launchpad.
  -r REPO, --repo REPO      Launchpad repo to dput to [default: pymor/unstable]
  -v REV, --version REV     force a version for the package if TAG is not usable
  -h, --help                Show this message.

"""

import sys
from docopt import docopt
from tempfile import TemporaryDirectory
from glob import glob
import os
import subprocess

POSSIBLE_DISTROS = {'xenial': '16.04', 'artful': '17.10', 'zesty': '17.04'}


def _check_call(args):
    subprocess.check_call(args)


def get_source(commitish):
    _check_call(['git', 'clone', 'https://github.com/pymor/pymor.git'])
    os.chdir('pymor')
    _check_call(['git', 'checkout', commitish])


def build_binary(distro):
    _check_call(['debocker', 'build', '--image', 'pymor/packaging_ubuntu:{}'.format(distro)])


def build_source_package(disto, numeric_id, commitish, rev):
    _check_call(['dch', '--force-distribution', '--distribution', distro, '-v',
        '{}~{}~ppa1'.format(rev, numeric_id), '-b', '"new upstream release"'])
    _check_call(['debuild', '-S', '-sa'])


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    commitish = args['TAG']
    upload = bool(args['--upload'])
    binary = bool(args['--binary'])
    repo = 'ppa:{}'.format(args['--repo'])
    rev = args['--version'] or commitish
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        get_source(commitish)

        for distro, numeric_id in POSSIBLE_DISTROS.items():
            _check_call(['git', 'reset', '--hard', commitish])
            _check_call(['git', 'clean', '-xdf'])

            build_source_package(distro, numeric_id, commitish, rev)
            if binary:
                build_binary(distro)
            if upload:
                files = glob('../*source.changes')
                try:
                    _check_call(['dput', '{}'.format(repo)] + [*files])
                    for fn in files:
                        os.remove(fn)
                except subprocess.CalledProcessError:
                    pass

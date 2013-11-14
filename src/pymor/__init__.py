# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.defaults import defaults

NO_VERSIONSTRING = '0.0.0-0-0'


def _make_version(string):
    pos = string.find('-')
    if pos == -1:
        string += '-0-0'
        pos = -4
    version = tuple(int(x) for x in string[:pos].split('.'))
    if pos > -1:
        git = string.strip().split('-')
        distance = int(git[1])
        shorthash = git[2]
        version = version + (distance, shorthash)
    return version

NO_VERSION = _make_version(NO_VERSIONSTRING)

try:
    import pymor.version
    revstring = pymor.version.revstring
except ImportError:
    import os.path
    import subprocess
    try:
        revstring = subprocess.check_output(['git', 'describe', '--tags', '--candidates', '20', '--match', '*.*.*'],
                                            cwd=os.path.dirname(__file__))
    except subprocess.CalledProcessError as e:
        import sys
        sys.stderr.write('''Warning: Could not determine current pyMOR version.
Failed to import pymor.version and 'git describe --tags --candidates 20 --match *.*.*'
returned

{}

(return code: {})
'''.format(e.output, e.returncode))
        revstring = NO_VERSIONSTRING
finally:
    version = _make_version(revstring)

VERSION = version
print('Loading pymor version {}'.format(VERSION))

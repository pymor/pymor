# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

NO_VERSIONSTRING = '0.0.0-0-0'

def _make_version(revstring):
    pos = revstring.find('-')
    version = tuple(int(x) for x in revstring[:pos].split('.'))
    if pos > -1:
        git = revstring.strip().split('-')
        distance = int(git[1])
        shorthash = git[2]
        version = version + (distance, shorthash)
    return version

NO_VERSION = _make_version(NO_VERSIONSTRING)

try:
    import pymor.version
    revstring = pymor.version.revstring
except ImportError:
    import subprocess
    try:
        revstring = subprocess.check_output(['git', 'describe', '--tags', '--candidates', '20', '--match', '*.*.*'])
    except:
        revstring = NO_VERSIONSTRING
finally:
    pos = revstring.find('-')
    version = tuple(int(x) for x in revstring[:pos].split('.'))
    if pos > -1:
        git = revstring.strip().split('-')
        distance = int(git[1])
        shorthash = git[2]
        version = version + (distance, shorthash)

VERSION = version
print('Loading pymor version {}'.format(VERSION))

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os

from pymor.core.defaults import load_defaults_from_file


class Version(object):
    def __init__(self, revstring):

        # special casing for debian versions like '0.1.3~precise~ppa9'
        if '~' in revstring:
            revstring = revstring[:revstring.index('~')]
        revstringparts = revstring.strip().split('-')
        if len(revstringparts) not in (1, 3):
            raise ValueError('Invalid revstring: ' + revstring)
        if len(revstringparts) == 3:
            self.distance = int(revstringparts[1])
            self.shorthash = revstringparts[2]
        else:
            self.distance = 0
            self.shorthash = ''

        version_parts = revstringparts[0].split('.')
        if version_parts[-1].find('rc') >= 0:
            s = version_parts[-1].split('rc')
            if len(s) != 2:
                raise ValueError('Invalid revstring')
            version_parts[-1] = s[0]
            self.rc_number = int(s[1])
            self.has_rc_number = True
        else:
            self.rc_number = 0
            self.has_rc_number = False

        self.version = tuple(int(x) for x in version_parts)
        self.full_version = self.version + (self.rc_number,)

    def __eq__(self, other):
        if not isinstance(other, Version):
            other = Version(other)
        return self.version == other.version and self.rc_number == other.rc_number and self.distance == other.distance

    def __lt__(self, other):
        if not isinstance(other, Version):
            other = Version(other)
        return self.full_version < other.full_version

    def __gt__(self, other):
        if not isinstance(other, Version):
            other = Version(other)
        return self.full_version > other.full_version

    def __str__(self):
        git_part = '-{}-{}'.format(self.distance, self.shorthash) if self.distance else ''
        version_part = '.'.join(map(str, self.version))
        rc_part = 'rc{}'.format(self.rc_number) if self.has_rc_number else ''
        return version_part + rc_part + git_part

    def __repr__(self):
        return 'Version({})'.format(str(self))


NO_VERSIONSTRING = '0.0.0-0-0'
NO_VERSION = Version(NO_VERSIONSTRING)

try:
    if 'PYMOR_DEB_VERSION' in os.environ:
        revstring = os.environ['PYMOR_DEB_VERSION']
    else:
        import pymor.version as _version

        revstring = getattr(_version, 'revstring', NO_VERSIONSTRING)
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
    VERSION = Version(revstring)

print('Loading pymor version {}'.format(VERSION))


import os
if 'PYMOR_DEFAULTS' in os.environ:
    filename = os.environ['PYMOR_DEFAULTS']
    if filename in ('', 'NONE'):
        print('Not loading any defaults from config file')
    else:
        for fn in filename.split(':'):
            if not os.path.exists(fn):
                raise IOError('Cannot load defaults from file ' + fn)
            print('Loading defaults from file ' + fn + ' (set by PYMOR_DEFAULTS)')
            load_defaults_from_file(fn)
else:
    filename = os.path.join(os.getcwd(), 'pymor_defaults.py')
    if os.path.exists(filename):
        if os.stat(filename).st_uid != os.getuid():
            raise IOError('Cannot load defaults from config file ' + filename
                          + ': not owned by user running Python interpreter')
        print('Loading defaults from file ' + filename)
        load_defaults_from_file(filename)

from pymor.core.logger import set_log_levels, set_log_format
set_log_levels()
set_log_format()

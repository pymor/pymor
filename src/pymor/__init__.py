# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


from pymor.defaults import defaults



class Version(object):

    def __init__(self, revstring):

        revstringparts = revstring.strip().split('-')
        if len(revstringparts) not in (1, 3):
            raise ValueError('Invalid revstring')
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
        else:
            self.rc_number = 0

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
        rc_part = 'rc{}'.format(self.rc_number) if self.rc_number else ''
        return version_part + rc_part + git_part

    def __repr__(self):
        return 'Version({})'.format(str(self))


NO_VERSIONSTRING = '0.0.0-0-0'
NO_VERSION = Version(NO_VERSIONSTRING)

try:
    import pymor.version as _version
    revstring = _version.revstring
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
    version = Version(revstring)

VERSION = version
print('Loading pymor version {}'.format(VERSION))

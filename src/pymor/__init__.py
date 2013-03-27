NO_VERSIONSTRING = '0.0.0-0'

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

NO_VERSION = (0,0,0,'NOVERSION')

try:
    import pymorversion
    version = pymorversion.version
except ImportError:
    try:
        import subprocess
        revstring = subprocess.check_output(['git', 'describe', '--tags', '--candidates', '20', '--match', '*.*.*'])
        pos = revstring.find('-')
        version = tuple(int(x) for x in revstring[:pos].split('.'))
        if pos > -1:
            git = revstring.strip().split('-')
            distance = int(git[1])
            shorthash = git[2]
            version = version + (distance, shorthash)
    except:
        version = NO_VERSION
finally:
    print('Loading pymor version {}'.format(version))
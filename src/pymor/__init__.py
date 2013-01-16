try:
    import pymorversion
    version = pymorversion.version
except ImportError:
    try:
        import subprocess
        revstring = subprocess.check_output(['git', 'describe', '--tags', '--candidates', '20', '--match', '*.*.*'])
        pos = revstring.find('-')
        version = tuple(revstring[:pos].split('.'))
        if pos > -1:
            git = revstring[pos+1:]
            version += tuple(git)
    except:
        version = (0,0,0,'NOVERSION')
finally:
    print('Loading pymor version {}'.format(version))
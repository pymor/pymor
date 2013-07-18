#!/usr/bin/env python

import os
import subprocess

import dependencies as deps

DEFAULT_RECIPE = {'system': ['echo make sure you have BLABLA installed'],
                  'local': deps.install_requires,
                  'venv_cmd': 'virtualenv'}
UBUNTU_RECIPE = {'system': ['sudo apt-get build-dep python-numpy',
                            'sudo apt-get install python-virtualenv'],
                  'local': deps.install_requires + deps.install_suggests,
                  'venv_cmd': '/usr/bin/virtualenv'}
DEFAULT_VENV_DIR = os.path.join(os.path.expandvars('$HOME'), 'virtualenv', 'pyMor')

def get_recipe():
    lsb_release = '/etc/lsb-release'
    if os.path.exists(lsb_release) and 'Ubuntu' in open(lsb_release).read():
        return UBUNTU_RECIPE
    return DEFAULT_RECIPE

if __name__ == '__main__':
    recipe = get_recipe()
    for cmd in recipe['system']:
        print('EXECUTING {}'.format(cmd))
        subprocess.check_call(cmd, shell=True)
    venvdir = DEFAULT_VENV_DIR
    print('VENCN ' + venvdir)
    subprocess.check_call([recipe['venv_cmd'], venvdir])
    activate = '. ' + os.path.join(venvdir, 'bin', 'activate')
    for cmd in ['pip install {}'.format(i) for i in recipe['local']] + ['python setup.py install']:
        print('EXECUTING {}'.format(cmd))
        subprocess.check_call('{} && {}'.format(activate, cmd), shell=True)


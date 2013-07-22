#!/usr/bin/env python

import os
import subprocess

import dependencies as deps

DEFAULT_RECIPE = {'name': 'default',
                  'system': ['echo make sure you have BLABLA installed'],
                  'local': deps.install_requires,
                  'venv_cmd': 'virtualenv'}
UBUNTU_12_04_RECIPE = {'name': 'Ubuntu 12.04',
                       'system': [  'sudo apt-get install build-essential cmake gfortran libqt4-dev libsuitesparse-dev '
                                  + 'libatlas-base-dev libfreetype6-dev libpng12-dev python-dev python-virtualenv '
                                  + 'python-pip python-tk tk-dev swig' ],
                       'local': deps.install_requires + deps.install_suggests,
                       'venv_cmd': '/usr/bin/virtualenv'}
DEFAULT_VENV_DIR = os.path.join(os.path.expandvars('$HOME'), 'virtualenv', 'pyMor')

def get_recipe():
    lsb_release = '/etc/lsb-release'
    if os.path.exists(lsb_release):
        release_description = open(lsb_release).read()
        if "Ubuntu 12.04" in release_description:
            return UBUNTU_12_04_RECIPE
        elif "Ubuntu" in release_description:
            warn('Unknown Ubuntu release, trying Ubuntu 12.04 recipe ...')
            return UBUNTU_12_04_RECIPE
    print('WARNING: Unknown platform, trying default recipe ...')
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

    print('''

Installation complete!

To use matplotlib with the Qt backend, create a file ~/.matplotlib/matplotlibrc
containing the lines

    backend      :  Qt4Agg
    backend.qt4  :  PySide

''')

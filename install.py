#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import subprocess
import time
import sys

import dependencies as deps

DEFAULT_RECIPE = {'name': 'default',
                  'system': ['echo make sure you have BLABLA installed'],
                  'local': deps.install_requires,
                  'venv_cmd': 'virtualenv'}
UBUNTU_12_04_RECIPE = {'name': 'Ubuntu 12.04',
                       'system': [  'sudo apt-get install build-essential cmake gfortran libqt4-dev libsuitesparse-dev '
                                  + 'libatlas-base-dev libfreetype6-dev libpng12-dev python2.7 python2.7-dev '
                                  +  'python2.7-tk python-pip python-virtualenv tk-dev swig' ],
                       'local': deps.install_requires + deps.install_suggests,
                       'venv_cmd': '/usr/bin/virtualenv'}

RECIPES = {'default': DEFAULT_RECIPE,
           'ubuntu_12_04': UBUNTU_12_04_RECIPE}

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
    parser = argparse.ArgumentParser(description='Installs pyMor with all its dependencies')
    parser.add_argument('--only-deps', action='store_true', help='install only dependencies')
    parser.add_argument('--recipe', choices=['default', 'ubuntu_12_04'],
                        help='installation recipe to use (otherwise auto-detected)')
    parser.add_argument('--without-python-path', action='store_true',
                        help='do not add pyMor to PYTHONPATH when --only-deps is used')
    parser.add_argument('--without-system-packages', action='store_true',
                        help='do not try to install required system packages')
    args = parser.parse_args()
    recipe = RECIPES[args.recipe] if args.recipe is not None else get_recipe()
    venvdir = DEFAULT_VENV_DIR
    if args.without_python_path and not args.only_deps:
        print('ERROR: --without-python-path can only be set when --only-deps is used')
        sys.exit(-1)
    if args.only_deps:
        pp = 'no' if args.without_python_path else 'yes'
    else:
        pp = 'n.a.'

    print('''
--------------------------------------------------------------------------------

About to install pyMor with the following configuration into a virtualenv:

    installation recipe:        {recipe}
    path of virtualenv:         {venvdir}
    install only dependencies:  {deps}
    install system packages:    {sys}
    add pyMor to PYTHONPATH:    {pp}

--------------------------------------------------------------------------------

'''.format(recipe=recipe['name'], venvdir=venvdir, deps='yes' if args.only_deps else 'no',
           pp=pp, sys='no' if args.without_system_packages else 'yes'))

    print('Staring installation in 5 seconds ', end='')
    sys.stdout.flush()
    for i in xrange(5):
        print('.', end='')
        sys.stdout.flush()
        time.sleep(1)
    print('\n\n')

    if not args.without_system_packages:
        for cmd in recipe['system']:
            print('EXECUTING {}'.format(cmd))
            subprocess.check_call(cmd, shell=True)

    print('VENCN --python=python2.7' + venvdir)
    subprocess.check_call([recipe['venv_cmd'], venvdir])
    activate = '. ' + os.path.join(venvdir, 'bin', 'activate')
    for cmd in ['pip install {}'.format(i) for i in recipe['local']]:
        print('EXECUTING {}'.format(cmd))
        subprocess.check_call('{} && {}'.format(activate, cmd), shell=True)

    if args.only_deps:
        print('Building pyMor C-extensions')
        subprocess.check_call('{} && python setup.py build_ext --inplace'.format(activate), shell=True)
        if not args.without_python_path:
            print('Adding pyMor to PYTHONPATH')
            with open(os.path.join(venvdir, 'lib/python2.7/site-packages/pymor.pth'), 'w') as f:
                    f.write(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src'))
    else:
        print('INSTALLING pyMor')
        subprocess.check_call('{} && python setup.py install'.format(activate), shell=True)

    print('''

Installation complete!

To use matplotlib with the Qt backend, create a file ~/.matplotlib/matplotlibrc
containing the lines

    backend      :  Qt4Agg
    backend.qt4  :  PySide

''')

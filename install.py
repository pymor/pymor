#!/usr/bin/env python2.7

from __future__ import print_function

import argparse
import os
import subprocess
import time
import sys

import dependencies as deps

DEFAULT_RECIPE = {'name': 'default',
                  'system': [  'echo make sure you have all build dependencies for numpy/scipy and Qt4 headers '
                             + 'installed' ],
                  'local': deps.install_requires + deps.install_suggests,
                  'venv_cmd': 'virtualenv'}
UBUNTU_12_04_RECIPE = {'name': 'Ubuntu 12.04',
                       'system': [  'sudo apt-get install build-essential cmake gfortran libqt4-dev libsuitesparse-dev '
                                  + 'libatlas-base-dev libfreetype6-dev libpng12-dev python2.7 python2.7-dev '
                                  +  'python2.7-tk python-pip python-virtualenv tk-dev swig' ],
                       'local': deps.install_requires + deps.install_suggests,
                       'venv_cmd': '/usr/bin/virtualenv'}
UBUNTU_13_04_RECIPE = {'name': 'Ubuntu 13.04',
                       'system': [  'sudo apt-get install build-essential cmake gfortran libqt4-dev libsuitesparse-dev '
                                  + 'libatlas-base-dev libfreetype6-dev libpng12-dev python2.7 python2.7-dev '
                                  + 'python2.7-tk python-pip python-virtualenv tk-dev swig' ],
                       'local': deps.install_requires + deps.install_suggests,
                       'venv_cmd': '/usr/bin/virtualenv'}

RECIPES = {'default': DEFAULT_RECIPE,
           'ubuntu_12_04': UBUNTU_12_04_RECIPE,
           'ubuntu_13_04': UBUNTU_13_04_RECIPE}

DEFAULT_VENV_DIR = os.path.join(os.path.expandvars('$HOME'), 'virtualenv', 'pyMor')

def print_separator():
    print('')
    print('-' * 80)
    print('')

def get_recipe():
    lsb_release = '/etc/lsb-release'
    if os.path.exists(lsb_release):
        release_description = open(lsb_release).read()
        if "Ubuntu 12.04" in release_description:
            return UBUNTU_12_04_RECIPE
        elif "Ubuntu 13.04" in release_description:
            return UBUNTU_13_04_RECIPE
        elif "Ubuntu" in release_description:
            print('Unknown Ubuntu release, trying Ubuntu 12.04 recipe ...')
            return UBUNTU_12_04_RECIPE
    print('WARNING: Unknown platform, trying default recipe ...')
    return DEFAULT_RECIPE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Installs pyMor with all its dependencies')
    parser.add_argument('--only-deps', action='store_true', help='install only dependencies')
    parser.add_argument('--recipe', choices=RECIPES.keys(),
                        help='installation recipe to use (otherwise auto-detected)')
    parser.add_argument('--virtualenv-dir', default=DEFAULT_VENV_DIR,
                        help='path of the virtualenv to be created')
    parser.add_argument('--without-python-path', action='store_true',
                        help='do not add pyMor to PYTHONPATH when --only-deps is used')
    parser.add_argument('--without-system-packages', action='store_true',
                        help='do not try to install required system packages')
    args = parser.parse_args()
    recipe = RECIPES[args.recipe] if args.recipe is not None else get_recipe()
    if args.without_python_path and not args.only_deps:
        print('ERROR: --without-python-path can only be set when --only-deps is used')
        sys.exit(-1)
    if args.only_deps:
        pp = 'no' if args.without_python_path else 'yes'
    else:
        pp = 'n.a.'
    venvdir = os.path.realpath(os.path.expanduser(args.virtualenv_dir))

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
        print_separator()
        print('Installing system packages\n')
        for cmd in recipe['system']:
            print('***** EXECUTING {}'.format(cmd))
            subprocess.check_call(cmd, shell=True)

    print_separator()
    print('***** CREATING VIRTUALENV\n')
    print('***** EXECUTING ' + recipe['venv_cmd'] + ' --python=python2.7 ' + venvdir)
    subprocess.check_call([recipe['venv_cmd'], '--python=python2.7', venvdir])
    activate = '. ' + os.path.join(venvdir, 'bin', 'activate')

    print_separator()
    print('***** Installing dependencies\n')
    for cmd in ['pip install {}'.format(i) for i in recipe['local']]:
        print('\n***** EXECUTING {}\n'.format(cmd))
        subprocess.check_call('{} && {}'.format(activate, cmd), shell=True)

    if args.only_deps:
        print_separator()
        print('***** BUILDING PYMOR C-EXTENSIONS\n')
        subprocess.check_call('{} && python setup.py build_ext --inplace'.format(activate), shell=True)
        if not args.without_python_path:
            print_separator()
            print('***** ADDING PYMOR TO PYTHONPATH\n')
            with open(os.path.join(venvdir, 'lib/python2.7/site-packages/pymor.pth'), 'w') as f:
                    f.write(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src'))
    else:
        print_separator()
        print('***** INSTALLING PYMOR\n')
        subprocess.check_call('{} && python setup.py install'.format(activate), shell=True)

    print('''

Installation complete!

To activate the pyMor virtualenv call

    source {}

To deactivate the virtualenv call

    deactivate

Matplotlib uses by default its Tk backend. To use the Qt backend, create a
file ~/.matplotlib/matplotlibrc containing the lines

    backend      :  Qt4Agg
    backend.qt4  :  PySide

'''.format(os.path.join(venvdir, 'bin/activate')))

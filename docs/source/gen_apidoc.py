# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import print_function

from types import FunctionType

import pkgutil

BUILD_DIR = 'generated'

CLASS_OPTIONS = [':show-inheritance:', ':members:', ':special-members:', ':exclude-members: __init__, __weakref__']
FUNCTION_OPTIONS = []
MODULE_OPTIONS = [':show-inheritance:']


def section(name, level=0, section_levels='*=-'):
    return name + '\n' + section_levels[level] * len(name) + '\n'


def walk(module):
    modules = []
    packages = []
    for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
        if ispkg:
            packages.append(module.__name__ + '.' + modname)
        else:
            modules.append(module.__name__ + '.' + modname)
    modules = sorted(modules)
    packages = sorted(packages)
    with open('{}/{}.rst'.format(BUILD_DIR, module.__name__), 'wb') as f:
        print(section('{} package'.format(module.__name__)), file=f)

        print('.. automodule:: ' + module.__name__, file=f)
        for option in MODULE_OPTIONS:
            print('    ' + option, file=f)
        print('', file=f)

        if packages:
            print(section('Subpackages', level=1), file=f)
            print('.. toctree::', file=f)
            for p in packages:
                print('    ' + p, file=f)
            print('', file=f)

        if modules:
            print(section('Submodules', level=1), file=f)
            for m in modules:
                print(section('{} module'.format(m.split('.')[-1]), level=2), file=f)
                print('.. automodule:: ' + m, file=f)
                for option in MODULE_OPTIONS:
                    print('    ' + option, file=f)
                print('', file=f)
                module = __import__(m, fromlist='none')
                for k, v in sorted(module.__dict__.iteritems()):
                    if isinstance(v, (type, FunctionType)) and v.__module__ == m:
                        if v.__name__.startswith('_') and not v.__doc__:
                            continue
                        print('---------\n\n', file=f)
                        if isinstance(v, type):
                            print('.. autoclass:: ' + m + '.' + k, file=f)
                            for option in CLASS_OPTIONS:
                                print('    ' + option, file=f)
                        else:
                            print('.. autofunction:: ' + m + '.' + k, file=f)
                            for option in FUNCTION_OPTIONS:
                                print('    ' + option, file=f)
                        print('', file=f)

    for packagename in packages:
        package = __import__(packagename, fromlist='none')
        walk(package)

#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import importlib
import pkgutil
import pymordemos
import sys
import runpy


def run():
    def _run(module):
        # only need to remove the modname from args, rest is automatic
        del sys.argv[1]
        runpy.run_module(module, init_globals=None, run_name='__main__', alter_sys=True)
        sys.exit(0)

    modules = []
    shorts = []
    fails = {}
    for _, module_name, _ in pkgutil.walk_packages(pymordemos.__path__, pymordemos.__name__ + '.'):
        short = module_name[len('pymordemos.'):]
        modules.append(module_name)
        shorts.append(short)
        try:
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as e:
            fails[short] = e

    def usage():
        msg = f'''Usage:
    {sys.argv[0]} DEMO_NAME | -h [DEMO_OPTIONS]

Arguments:
    -h           this message
    DEMO_NAME    select one from these: {",".join(shorts)}
    DEMO_OPTIONS any arguments for the demo, including -h for detailed help
'''
        print(msg)
        if len(fails):
            print('\nThere are some pyMOR demos for which additional packages need to be installed:')
            print('\t'+'\n\t'.join(fails))
            print('\nYou can try to `pip install pymor[full]` to install optional dependencies.\n')
        sys.exit(0)

    if len(sys.argv) < 2:
        usage()
    demo = sys.argv[1]
    if demo in shorts:
        if demo in fails.keys():
            print(str(fails[demo]))
            print(f'\nThe {demo} pyMOR demo needs additional packages to be installed (see above error for details).')
            print('\nYou can try to `pip install pymor[full]` to install optional dependencies.\n')
            sys.exit(-1)
        _run(modules[shorts.index(demo)])
    usage()


if __name__ == '__main__':
    run()

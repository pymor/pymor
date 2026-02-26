# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from shutil import rmtree

from cyclopts.docs import generate_rst_docs

import pymordemos

OUT_DIR = Path(__file__).parent / 'source' / 'demos'
rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir()

demos = [m.name for m in iter_modules(pymordemos.__path__)]

with open(OUT_DIR / 'demos.rst', 'w') as index:
    print("""
************
Demo Scripts
************

.. toctree::
   :maxdepth: 1

"""[1:-1], file=index)
    for demo in demos:
        app = import_module('pymordemos.' + demo).app
        app._name = 'pymor-demo ' + demo
        with open(OUT_DIR / f'{demo}.rst', 'w') as f:
            docs = generate_rst_docs(app, generate_toc=False).splitlines()
            print('\n'.join(docs[:2]), file=f)
            print('=' * len(demo), file=f)
            print(demo, file=f)
            print('=' * len(demo), file=f)
            print('\n'.join(docs[5:]), file=f)
        print('   ' + demo, file=index)

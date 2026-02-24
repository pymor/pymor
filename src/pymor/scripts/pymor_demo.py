# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
from importlib import import_module
from pkgutil import iter_modules
from typing import Annotated, Literal

from cyclopts import App
from cyclopts.types import Parameter

import pymordemos
from pymor.core.exceptions import DependencyMissingError

demos = tuple(m.name for m in iter_modules(pymordemos.__path__))

app = App(help_on_error=True)

@app.default
def main(demo: Literal[demos], /, *args: Annotated[str, Parameter(allow_leading_hyphen=True)]):
    """Runs a pyMOR demo script.

    Parameters
    ----------
    demo
        Name of the demo script to run.
    args
        Arguments for the demo script.
    """
    app = import_module('pymordemos.' + demo).app
    app._name = 'pymor-demo ' + demo
    try:
        app(args)
    except DependencyMissingError as e:
        print(f"""

------------------------------------------------------------
DEPENDENCY MISSING

An optional depenency that is needed to run this demo script
could not be found!

Missing dependency: {e.dependency}
------------------------------------------------------------

"""[1:-1])
        sys.exit(1)


if __name__ == '__main__':
    app()

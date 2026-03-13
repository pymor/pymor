# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pathlib import Path

from cyclopts import App

from pymor.core.cache import print_cached_methods
from pymor.core.config import config
from pymor.core.defaults import print_defaults, write_defaults_to_file

app = App(help_on_error=True)


@app.command
def write_defaults(filename: Path = Path('./pymor_defaults.py')):
    """Write pyMOR defaults to a config file.

    pyMOR will look for a file 'pymor_defaults.py' in the
    current directory and automatically load defaults from this
    file. Alternatively, defaults can be loaded using
    `load_defaults_from_file` or by setting the `PYMOR_DEFAULTS`
    environment variable.

    Parameters
    ----------
    filename
        Write defaults to this file.
    """
    write_defaults_to_file(str(filename))


@app.default
def main(*, show_defaults: bool = False, show_cached_methods: bool = False, all: bool = False):
    """Show pyMOR config.

    Parameters
    ----------
    show_defaults
        Also show values of all pyMOR defaults.
    show_cached_methods
        Also show which methods are cached by default.
    all
        Print all information.
    """
    print(config)

    if show_defaults or all:
        print_defaults()

    if show_cached_methods or all:
        print_cached_methods()


if __name__ == '__main__':
    app()

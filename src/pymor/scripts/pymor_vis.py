#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os

import typer

from pymor.core.pickle import load


def main(filename: str, delete: bool = typer.Option(False, help='Delete file when done.')):
    try:
        with open(filename, 'rb') as f:
            data = load(f)

        assert 'dim' in data
        assert 'block' not in data
        dim = data.pop('dim')

        if dim == 1:
            from pymor.discretizers.builtin.gui.qt import visualize_matplotlib_1d
            visualize_matplotlib_1d(block=True, **data)
        else:
            from pymor.discretizers.builtin.gui.qt import visualize_patch
            visualize_patch(block=True, **data)
    finally:
        if delete:
            os.remove(filename)


def run():
    typer.run(main)


if __name__ == '__main__':
    run()

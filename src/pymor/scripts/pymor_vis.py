# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os

from cyclopts import App

from pymor.core.pickle import load

app = App(help_on_error=True)

@app.default
def main(filename: str, delete: bool = False):
    """Visualize pickled data from built-in discretization toolkit.

    The pickled data has to be a dict of visualizer arguments
    (grid, |VectorArray|, further options) along with a `dim` key,
    which is used to select the visualizer to be used.

    Parameters
    ----------
    filename
        The pickled data.
    delete
        Delete file when done.
    """
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


if __name__ == '__main__':
    app()

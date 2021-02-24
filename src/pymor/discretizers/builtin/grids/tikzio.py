# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.config import config
from pymor.discretizers.builtin.grids.rect import RectGrid


def write_tikz(grid, data, filename, codim=2, colorbar=True,
               use_subplots=True, cols=2):
    """Output grid-associated data as a PGFPlots figure.

    Parameters
    ----------
    grid
        Must be a |RectGrid| at the moment.
    data
        |VectorArray| with either cell (ie one datapoint per codim 0 entity)
        or vertex (ie one datapoint per codim 2 entity) data in each array element.
    filename
        Name of main output file.
    codim
        The codimension associated with the data.
    colorbar
        If `True` include a colorbar in the plot.
    use_subplots
        Use subplots to visualize a time series.
    cols
        Number of colums when subplots are used.
    """
    if not config.HAVE_MATPLOTLIB:
        raise ImportError('matplotlib missing')
    if not config.HAVE_TIKZPLOTLIB:
        raise ImportError('tikzplotlib missing')
    if not isinstance(grid, RectGrid):
        raise NotImplementedError
    if grid.identify_bottom_top or grid.identify_left_right:
        raise NotImplementedError
    if codim not in (0, 2):
        raise NotImplementedError

    if len(data) == 0:
        raise ValueError('Nothing to visualize')
    assert (codim == 0 and data.dim == grid.size(0)) or \
           (codim == 2 and data.dim == grid.size(2))

    from matplotlib import pyplot as plt
    import tikzplotlib

    X = np.linspace(grid.domain[0][0], grid.domain[1][0], grid.num_intervals[0] + 1)
    Y = np.linspace(grid.domain[0][1], grid.domain[1][1], grid.num_intervals[1] + 1)

    def write_one_figure(filename, data):
        fig = plt.figure()
        if codim == 0:
            data = data.to_numpy().reshape((grid.num_intervals[1], grid.num_intervals[0]))
            plt.pcolormesh(X, Y, data, shading='flat', figure=fig)
        elif codim == 2:
            data = data.to_numpy().reshape((grid.num_intervals[1] + 1, grid.num_intervals[0] + 1))
            plt.pcolormesh(X, Y, data, shading='gouraud', figure=fig)
        else:
            assert False
        if colorbar:
            plt.colorbar()
        tikzplotlib.save(filename, override_externals=True)
        plt.close(fig)

    def write_subplots(filename, data):
        fig = plt.figure()
        rows = len(data)//cols + (1 if len(data) % cols != 0 else 0)
        c_min = np.min(data.to_numpy())
        c_max = np.max(data.to_numpy())
        for i in range(len(data)):
            plt.subplot(rows, cols, i+1)
            if codim == 0:
                dat = data[i].to_numpy().reshape((grid.num_intervals[1], grid.num_intervals[0]))
                plt.pcolormesh(X, Y, dat, shading='flat', figure=fig)
            elif codim == 2:
                dat = data[i].to_numpy().reshape((grid.num_intervals[1] + 1, grid.num_intervals[0] + 1))
                plt.pcolormesh(X, Y, dat, shading='gouraud', figure=fig)
            else:
                assert False
            plt.clim(c_min, c_max)
            if colorbar and i+1 == len(data):
                plt.colorbar()
        tikzplotlib.save(filename, override_externals=True)
        plt.close(fig)

    if len(data) == 1:
        write_one_figure(filename, data)
    else:
        if use_subplots:
            write_subplots(filename, data)
        else:
            filename_base = filename[:-4] if filename.endswith('.tex') else filename
            for i, d in enumerate(data):
                write_one_figure(f'{filename_base}_{i:03}.tex', d)

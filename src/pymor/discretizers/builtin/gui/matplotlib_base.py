# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Visualization of grid data using matplotlib.

This module provides widgets for displaying plots of
scalar data assigned to one- and two-dimensional grids using
:mod:`matplotlib`. These widgets are not intended to be used directly.
"""
from pymor.core.config import config

config.require('MATPLOTLIB')

import numpy as np

from pymor.discretizers.builtin.grids.constructions import flatten_grid
from pymor.discretizers.builtin.grids.referenceelements import square, triangle


class MatplotlibPatchAxes:

    def __init__(self, ax, grid, bounding_box=None, codim=2):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)

        subentities, coordinates, entity_map = flatten_grid(grid)
        self.subentities = subentities if grid.reference_element is triangle \
            else np.vstack((subentities[:, 0:3], subentities[:, [2, 3, 0]]))
        self.reference_element = grid.reference_element
        self.animate = self.set

        if bounding_box is None:
            bounding_box = grid.bounding_box()
        assert len(bounding_box) == 2 and all(len(b) == 2 for b in bounding_box)
        aspect_ratio = (bounding_box[1][1] - bounding_box[0][1]) / (bounding_box[1][0] - bounding_box[0][0])

        self.codim = codim

        if codim == 2:
            self.p = ax.tripcolor(coordinates[:, 0], coordinates[:, 1], np.zeros(len(coordinates)),
                                  triangles=subentities, shading='gouraud')
        else:
            self.p = ax.tripcolor(coordinates[:, 0], coordinates[:, 1], facecolors=np.zeros(len(subentities)),
                                  triangles=subentities, shading='flat')

        # thin plots look ugly with a huge colorbar on the right
        if aspect_ratio < 0.75:
            orientation = 'horizontal'
        else:
            orientation = 'vertical'
        self.cbar = ax.figure.colorbar(self.p, ax=ax, orientation=orientation)

    def set(self, U, vmin, vmax):
        if self.codim == 2:
            self.p.set_array(U)
        elif self.reference_element is triangle:
            self.p.set_array(U)
        else:
            self.p.set_array(np.tile(U, 2))
        self.p.set_clim(vmin, vmax)
        self.cbar.mappable.set_clim(vmin, vmax)
        return (self.p,)


class Matplotlib1DAxes:

    def __init__(self, figure, grid, count, legend=None, codim=1, separate_plots=False,
                 columns=2):
        self.codim = codim

        if separate_plots:
            rows = int(np.ceil(count / columns))
            self.ax = figure.subplots(rows, columns, squeeze=False).flatten()
            for ax in self.ax[count:]:
                ax.set_axis_off()
        else:
            self.ax = (figure.gca(),)
        self.codim = codim

        centers = grid.centers(1)
        if grid.identify_left_right:
            centers = np.concatenate((centers, [[grid.domain[1]]]), axis=0)
            self.periodic = True
        else:
            self.periodic = False
        if self.codim == 1:
            xs = centers
        else:
            xs = np.repeat(centers, 2)[1:-1]
        if separate_plots:
            self.lines = [ax.plot(xs, np.zeros_like(xs))[0] for ax in self.ax[:count]]
        else:
            self.lines = [self.ax[0].plot(xs, np.zeros_like(xs))[0] for _ in range(count)]
        if legend:
            if separate_plots:
                for ax, l in zip(self.ax, legend):
                    ax.legend([l])
            else:
                self.ax[0].legend(legend)

    def set(self, U, vmin, vmax):
        for i, u in enumerate(U):
            if self.codim == 1:
                if self.periodic:
                    self.lines[i].set_ydata(np.concatenate((u, [u[0]])))
                else:
                    self.lines[i].set_ydata(u)
            else:
                self.lines[i].set_ydata(np.repeat(u, 2))
        for ax, mi, ma in zip(self.ax, vmin, vmax):
            pad = (ma - mi) * 0.1
            ax.set_ylim(mi - pad, ma + pad)

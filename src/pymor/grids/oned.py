#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.grids.interfaces import AffineGridWithOrthogonalCentersInterface
from pymor.grids.referenceelements import line


class OnedGrid(AffineGridWithOrthogonalCentersInterface):
    """One-dimensional |Grid| on an interval.

    Parameters
    ----------
    domain
        Tuple `(left, right)` containing the left and right boundary of the domain.
    num_intervals
        The number of codim-0 entities.
    """

    dim = 1
    dim_outer = 1
    reference_element = line

    def __init__(self, domain=(0, 1), num_intervals=4, identify_left_right=False):
        assert domain[0] < domain[1]
        self.reference_element = line
        self._domain = np.array(domain)
        self._num_intervals = num_intervals
        self._identify_left_right = identify_left_right
        self._sizes = [num_intervals, num_intervals] if identify_left_right else [num_intervals, num_intervals + 1]
        self._width = np.abs(self._domain[1] - self._domain[0]) / self._num_intervals
        self.__subentities = np.vstack((np.arange(self._num_intervals, dtype=np.int32),
                                        np.arange(self._num_intervals, dtype=np.int32) + 1))
        if identify_left_right:
            self.__subentities[-1, -1] = 0
        self.__A = np.ones(self._num_intervals, dtype=np.int32)[:, np.newaxis, np.newaxis] * self._width
        self.__B = (self._domain[0] + self._width * (np.arange(self._num_intervals, dtype=np.int32)))[:, np.newaxis]

    def __reduce__(self):
        return (OnedGrid,
                (self._domain, self._num_intervals, self._identify_left_right))

    def __str__(self):
        return ('OnedGrid, domain [{xmin},{xmax}]'
                + ', {elements} elements'
                + ', {vertices} vertices'
                ).format(xmin=self._domain[0],
                         xmax=self._domain[1],
                         elements=self.size(0),
                         vertices=self.size(1))

    def size(self, codim=0):
        assert 0 <= codim <= 1, 'codim has to be between 0 and {}!'.format(self.dim)
        return self._sizes[codim]

    def subentities(self, codim, subentity_codim):
        assert 0 <= codim <= 1, 'Invalid codimension'
        assert codim <= subentity_codim <= self.dim, 'Invalid subentity codimension'
        if codim == 0:
            if subentity_codim == 0:
                return np.arange(self.size(0), dtype='int32')[:, np.newaxis]
            else:
                return self.__subentities.T
        else:
            return super().subentities(codim, subentity_codim)

    def embeddings(self, codim):
        if codim == 0:
            return self.__A, self.__B
        else:
            return super().embeddings(codim)

    def bounding_box(self):
        return np.array(self._domain).reshape((2, 1))

    def orthogonal_centers(self):
        return self.centers(0)

    def visualize(self, U, codim=2, **kwargs):
        """Visualize scalar data associated to the grid as a patch plot.

        Parameters
        ----------
        U
            |NumPy array| of the data to visualize. If `U.dim == 2 and len(U) > 1`, the
            data is visualized as a time series of plots. Alternatively, a tuple of
            |Numpy arrays| can be provided, in which case a subplot is created for
            each entry of the tuple. The lengths of all arrays have to agree.
        codim
            The codimension of the entities the data in `U` is attached to (either 0 or 2).
        kwargs
            See :func:`~pymor.gui.qt.visualize_patch`
        """
        from pymor.gui.qt import visualize_matplotlib_1d
        from pymor.vectorarrays.interfaces import VectorArrayInterface
        from pymor.vectorarrays.numpy import NumpyVectorArray
        if isinstance(U, (np.ndarray, VectorArrayInterface)):
            U = (U,)
        assert all(isinstance(u, (np.ndarray, VectorArrayInterface)) for u in U)
        U = tuple(NumpyVectorArray(u) if isinstance(u, np.ndarray) else
                  u if isinstance(u, NumpyVectorArray) else
                  NumpyVectorArray(u.data)
                  for u in U)
        visualize_matplotlib_1d(self, U, codim=codim, **kwargs)

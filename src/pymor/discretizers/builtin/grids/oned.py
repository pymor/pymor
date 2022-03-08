#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.discretizers.builtin.grids.interfaces import GridWithOrthogonalCenters
from pymor.discretizers.builtin.grids.referenceelements import line


class OnedGrid(GridWithOrthogonalCenters):
    """One-dimensional |Grid| on an interval.

    Parameters
    ----------
    domain
        Tuple `(left, right)` containing the left and right boundary of the domain.
    num_intervals
        The number of codim-0 entities.
    """

    dim = 1
    reference_element = line

    def __init__(self, domain=(0, 1), num_intervals=4, identify_left_right=False):
        domain = np.array(domain)
        assert domain.ndim == 1
        assert domain[0] < domain[1]
        self.__auto_init(locals())
        self._sizes = [num_intervals, num_intervals] if identify_left_right else [num_intervals, num_intervals + 1]
        self._width = np.abs(self.domain[1] - self.domain[0]) / self.num_intervals
        self.__subentities = np.vstack((np.arange(self.num_intervals, dtype=np.int32),
                                        np.arange(self.num_intervals, dtype=np.int32) + 1))
        if identify_left_right:
            self.__subentities[-1, -1] = 0
        self.__A = np.ones(self.num_intervals, dtype=np.int32)[:, np.newaxis, np.newaxis] * self._width
        self.__B = (self.domain[0] + self._width * (np.arange(self.num_intervals, dtype=np.int32)))[:, np.newaxis]

    def __reduce__(self):
        return (OnedGrid,
                (self.domain, self.num_intervals, self.identify_left_right))

    def __str__(self):
        return (f'OnedGrid, domain [{self.domain[0]},{self.domain[1]}], '
                f'{self.size(0)} elements, {self.size(1)} vertices')

    def size(self, codim=0):
        assert 0 <= codim <= 1, f'codim has to be between 0 and {self.dim}!'
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
        return np.array(self.domain).reshape((2, 1))

    def orthogonal_centers(self):
        return self.centers(0)

    def visualize(self, U, codim=1, **kwargs):
        """Visualize scalar data associated to the grid as a patch plot.

        Parameters
        ----------
        U
            |NumPy array| of the data to visualize. If `U.dim == 2 and len(U) > 1`, the
            data is visualized as a time series of plots. Alternatively, a tuple of
            |Numpy arrays| can be provided, in which case a subplot is created for
            each entry of the tuple. The lengths of all arrays have to agree.
        codim
            The codimension of the entities the data in `U` is attached to (either 0 or 1).
        kwargs
            See :func:`~pymor.discretizers.builtin.gui.visualizers.OnedVisualizer.visualize`
        """
        from pymor.discretizers.builtin.gui.visualizers import OnedVisualizer
        from pymor.vectorarrays.interface import VectorArray
        from pymor.vectorarrays.numpy import NumpyVectorSpace
        if isinstance(U, (np.ndarray, VectorArray)):
            U = (U,)
        assert all(isinstance(u, (np.ndarray, VectorArray)) for u in U)
        U = tuple(NumpyVectorSpace.make_array(u) if isinstance(u, np.ndarray) else
                  u if isinstance(u.space, NumpyVectorSpace) else
                  NumpyVectorSpace.make_array(u.to_numpy())
                  for u in U)
        OnedVisualizer(self, codim=codim).visualize(U, **kwargs)

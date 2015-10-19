# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.cache import cached
from pymor.grids.interfaces import AffineGridWithOrthogonalCentersInterface
from pymor.grids.referenceelements import triangle


class TriaGrid(AffineGridWithOrthogonalCentersInterface):
    """Basic implementation of a triangular grid on a rectangular domain.

    The global face, edge and vertex indices are given as follows ::

                 6---------10----------7---------11----------8
                 | \                 / | \                 / |
                 |    22   10    18    |    23   11    19    |
                 |       \     /       |       \     /       |
                 3    14   11     6    4    15   12     7    5
                 |       /     \       |       /     \       |
                 |    14    2    26    |    15    3    27    |
                 | /                 \ | /                 \ |
                 3----------8----------4----------9----------5
                 | \                 / | \                 / |
                 |    20    8    16    |    21    9    17    |
                 |       \     /       |       \     /       |
                 0    12    9     4    1    13   10     5    2
                 |       /     \       |       /     \       |
                 |    12    0    24    |    13    1    25    |
                 | /                 \ | /                 \ |
                 0----------6----------1----------7----------2

    Parameters
    ----------
    num_intervals
        Tuple `(n0, n1)` determining a grid with `n0` x `n1` codim-0 entities.
    domain
        Tuple `(ll, ur)` where `ll` defines the lower left and `ur` the upper right
        corner of the domain.
    """

    dim = 2
    dim_outer = 2
    reference_element = triangle

    def __init__(self, num_intervals=(2, 2), domain=([0, 0], [1, 1]),
                 identify_left_right=False, identify_bottom_top=False):
        if identify_left_right:
            assert num_intervals[0] > 1
        if identify_bottom_top:
            assert num_intervals[1] > 1
        self.num_intervals = num_intervals
        self.domain = np.array(domain)
        self.identify_left_right = identify_left_right
        self.identify_bottom_top = identify_bottom_top

        self.x0_num_intervals = x0_num_intervals = num_intervals[0]
        self.x1_num_intervals = x1_num_intervals = num_intervals[1]
        self.x0_range = self.domain[:, 0]
        self.x1_range = self.domain[:, 1]
        self.x0_width = self.x0_range[1] - self.x0_range[0]
        self.x1_width = self.x1_range[1] - self.x1_range[0]
        self.x0_diameter = self.x0_width / x0_num_intervals
        self.x1_diameter = self.x1_width / x1_num_intervals
        n_elements = x0_num_intervals * x1_num_intervals * 4

        # TOPOLOGY
        n_outer_vertices = (x0_num_intervals + 1 - identify_left_right) * (x1_num_intervals + 1 - identify_bottom_top)
        self.__sizes = (n_elements,
                        ((x0_num_intervals + 1 - identify_left_right) * x1_num_intervals +
                         (x1_num_intervals + 1 - identify_bottom_top) * x0_num_intervals +
                         n_elements),
                        n_outer_vertices + int(n_elements / 4))

        # calculate subentities -- codim-1
        V_EDGE_H_INDICES = np.arange(x0_num_intervals + 1, dtype=np.int32)
        if identify_left_right:
            V_EDGE_H_INDICES[-1] = 0
        V_EDGE_V_INDICES = np.arange(x1_num_intervals, dtype=np.int32) * (x0_num_intervals + 1 - identify_left_right)
        V_EDGE_INDICES = V_EDGE_V_INDICES[:, np.newaxis] + V_EDGE_H_INDICES
        num_v_edges = x1_num_intervals * (x0_num_intervals + 1 - identify_left_right)

        H_EDGE_H_INDICES = np.arange(x0_num_intervals, dtype=np.int32)
        H_EDGE_V_INDICES = np.arange(x1_num_intervals + 1, dtype=np.int32)
        if identify_bottom_top:
            H_EDGE_V_INDICES[-1] = 0
        H_EDGE_V_INDICES *= x0_num_intervals
        H_EDGE_INDICES = H_EDGE_V_INDICES[:, np.newaxis] + H_EDGE_H_INDICES + num_v_edges
        num_h_edges = x0_num_intervals * (x1_num_intervals + 1 - identify_bottom_top)

        D_EDGE_LL_INDICES = np.arange(n_elements / 4, dtype=np.int32) + (num_v_edges + num_h_edges)
        D_EDGE_UR_INDICES = D_EDGE_LL_INDICES + int(n_elements / 4)
        D_EDGE_UL_INDICES = D_EDGE_UR_INDICES + int(n_elements / 4)
        D_EDGE_LR_INDICES = D_EDGE_UL_INDICES + int(n_elements / 4)

        E0 = np.array([H_EDGE_INDICES[:-1, :].ravel(),
                       D_EDGE_LR_INDICES,
                       D_EDGE_LL_INDICES]).T
        E1 = np.array([V_EDGE_INDICES[:, 1:].ravel(),
                       D_EDGE_UR_INDICES,
                       D_EDGE_LR_INDICES]).T
        E2 = np.array([H_EDGE_INDICES[1:, :].ravel(),
                       D_EDGE_UL_INDICES,
                       D_EDGE_UR_INDICES]).T
        E3 = np.array([V_EDGE_INDICES[:, :-1].ravel(),
                       D_EDGE_LL_INDICES,
                       D_EDGE_UL_INDICES]).T

        codim1_subentities = np.vstack((E0, E1, E2, E3))

        # calculate subentities -- codim-2
        VERTEX_H_INDICES = np.arange(x0_num_intervals + 1, dtype=np.int32)
        if identify_left_right:
            VERTEX_H_INDICES[-1] = 0
        VERTEX_V_INDICES = np.arange(x1_num_intervals + 1, dtype=np.int32)
        if identify_bottom_top:
            VERTEX_V_INDICES[-1] = 0
        VERTEX_V_INDICES *= x0_num_intervals + 1 - identify_left_right
        VERTEX_NUMERS = VERTEX_V_INDICES[:, np.newaxis] + VERTEX_H_INDICES
        VERTEX_CENTER_NUMBERS = np.arange(x0_num_intervals * x1_num_intervals, dtype=np.int32) + n_outer_vertices

        V0 = np.array([VERTEX_CENTER_NUMBERS,
                       VERTEX_NUMERS[:-1, :-1].ravel(),
                       VERTEX_NUMERS[:-1, 1:].ravel()]).T
        V1 = np.array([VERTEX_CENTER_NUMBERS,
                       VERTEX_NUMERS[:-1, 1:].ravel(),
                       VERTEX_NUMERS[1:, 1:].ravel()]).T
        V2 = np.array([VERTEX_CENTER_NUMBERS,
                       VERTEX_NUMERS[1:, 1:].ravel(),
                       VERTEX_NUMERS[1:, :-1].ravel()]).T
        V3 = np.array([VERTEX_CENTER_NUMBERS,
                       VERTEX_NUMERS[1:, :-1].ravel(),
                       VERTEX_NUMERS[:-1, :-1].ravel()]).T

        codim2_subentities = np.vstack((V0, V1, V2, V3))
        self.__subentities = (codim1_subentities, codim2_subentities)

        # GEOMETRY

        # embeddings
        x0_shifts = np.arange(x0_num_intervals) * self.x0_diameter + (self.x0_range[0] + 0.5 * self.x0_diameter)
        x1_shifts = np.arange(x1_num_intervals) * self.x1_diameter + (self.x1_range[0] + 0.5 * self.x1_diameter)
        B = np.tile(np.array(np.meshgrid(x0_shifts, x1_shifts)).reshape((2, -1)).T,
                    (4, 1))

        ROT45  = np.array([[1./np.sqrt(2.),   -1./np.sqrt(2.)],
                           [1./np.sqrt(2.),    1./np.sqrt(2.)]])
        ROT135 = np.array([[-1./np.sqrt(2.),  -1./np.sqrt(2.)],
                           [1./np.sqrt(2.),   -1./np.sqrt(2.)]])
        ROT225 = np.array([[-1./np.sqrt(2.),   1./np.sqrt(2.)],
                           [-1./np.sqrt(2.),  -1./np.sqrt(2.)]])
        ROT315 = np.array([[1./np.sqrt(2.),    1./np.sqrt(2.)],
                           [-1./np.sqrt(2.),   1./np.sqrt(2.)]])
        SCAL = np.diag([self.x0_diameter / np.sqrt(2), self.x1_diameter / np.sqrt(2)])
        A0 = np.tile(SCAL.dot(ROT225), (n_elements / 4, 1, 1))
        A1 = np.tile(SCAL.dot(ROT315), (n_elements / 4, 1, 1))
        A2 = np.tile(SCAL.dot(ROT45), (n_elements / 4, 1, 1))
        A3 = np.tile(SCAL.dot(ROT135), (n_elements / 4, 1, 1))
        A = np.vstack((A0, A1, A2, A3))
        self.__embeddings = (A, B)

    def __reduce__(self):
        return (TriaGrid,
                (self.num_intervals, self.domain, self.identify_left_right, self.identify_bottom_top))

    def __str__(self):
        return (('Tria-Grid on domain [{xmin},{xmax}] x [{ymin},{ymax}]\n' +
                 'x0-intervals: {x0ni}, x1-intervals: {x1ni}\n' +
                 'elements: {elements}, edges: {edges}, vertices: {vertices}')
                .format(xmin=self.x0_range[0], xmax=self.x0_range[1],
                        ymin=self.x1_range[0], ymax=self.x1_range[1],
                        x0ni=self.x0_num_intervals, x1ni=self.x1_num_intervals,
                        elements=self.size(0), edges=self.size(1), vertices=self.size(2)))

    def size(self, codim=0):
        assert 0 <= codim <= 2, 'Invalid codimension'
        return self.__sizes[codim]

    def subentities(self, codim, subentity_codim):
        assert 0 <= codim <= 2, 'Invalid codimension'
        assert codim <= subentity_codim <= 2, 'Invalid subentity codimension'
        if codim == 0:
            if subentity_codim == 0:
                return np.arange(self.size(0), dtype='int32')[:, np.newaxis]
            else:
                return self.__subentities[subentity_codim - 1]
        else:
            return super(TriaGrid, self).subentities(codim, subentity_codim)

    def embeddings(self, codim=0):
        if codim == 0:
            return self.__embeddings
        else:
            return super(TriaGrid, self).embeddings(codim)

    def bounding_box(self):
        return np.array(self.domain)

    @cached
    def orthogonal_centers(self):
        embeddings = self.embeddings(0)
        ne4 = len(embeddings[0]) / 4
        if self.x0_diameter > self.x1_diameter:
            x0_fac = (self.x1_diameter / 2) ** 2 / (3 * (self.x0_diameter / 2) ** 2)
            x1_fac = 1./3.
        else:
            x1_fac = (self.x0_diameter / 2) ** 2 / (3 * (self.x1_diameter / 2) ** 2)
            x0_fac = 1./3.
        C0 = embeddings[0][:ne4].dot(np.array([x1_fac, x1_fac])) + embeddings[1][:ne4]
        C1 = embeddings[0][ne4:2*ne4].dot(np.array([x0_fac, x0_fac])) + embeddings[1][ne4:2*ne4]
        C2 = embeddings[0][2*ne4:3*ne4].dot(np.array([x1_fac, x1_fac])) + embeddings[1][2*ne4:3*ne4]
        C3 = embeddings[0][3*ne4:4*ne4].dot(np.array([x0_fac, x0_fac])) + embeddings[1][3*ne4:4*ne4]
        return np.concatenate((C0, C1, C2, C3), axis=0)

    def visualize(self, U, codim=2, **kwargs):
        """Visualize scalar data associated to the grid as a patch plot.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case a subplot is created for each entry of the tuple. The
            lengths of all arrays have to agree.
        codim
            The codimension of the entities the data in `U` is attached to (either 0 or 2).
        kwargs
            See :func:`~pymor.gui.qt.visualize_patch`
        """
        from pymor.gui.qt import visualize_patch
        from pymor.vectorarrays.numpy import NumpyVectorArray
        if not isinstance(U, NumpyVectorArray):
            U = NumpyVectorArray(U, copy=False)
        bounding_box = kwargs.pop('bounding_box', self.domain)
        visualize_patch(self, U, codim=codim, bounding_box=bounding_box, **kwargs)

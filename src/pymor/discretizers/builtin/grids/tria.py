# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.cache import cached
from pymor.discretizers.builtin.grids.interfaces import GridWithOrthogonalCenters
from pymor.discretizers.builtin.grids.referenceelements import triangle


class TriaGrid(GridWithOrthogonalCenters):
    r"""Basic implementation of a triangular grid on a rectangular domain.

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
    identify_left_right
        If `True`, the left and right boundaries are identified, i.e. the left-most
        codim-0 entities become neighbors of the right-most codim-0 entities.
    identify_bottom_top
        If `True`, the bottom and top boundaries are identified, i.e. the bottom-most
        codim-0 entities become neighbors of the top-most codim-0 entities.
    """

    dim = 2
    reference_element = triangle

    def __init__(self, num_intervals=(2, 2), domain=([0, 0], [1, 1]),
                 identify_left_right=False, identify_bottom_top=False):
        if identify_left_right:
            assert num_intervals[0] > 1
        if identify_bottom_top:
            assert num_intervals[1] > 1
        domain = np.array(domain)
        self.__auto_init(locals())

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
                        ((x0_num_intervals + 1 - identify_left_right) * x1_num_intervals
                         + (x1_num_intervals + 1 - identify_bottom_top) * x0_num_intervals
                         + n_elements),
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
        A0 = np.tile(SCAL.dot(ROT225), (int(n_elements / 4), 1, 1))
        A1 = np.tile(SCAL.dot(ROT315), (int(n_elements / 4), 1, 1))
        A2 = np.tile(SCAL.dot(ROT45), (int(n_elements / 4), 1, 1))
        A3 = np.tile(SCAL.dot(ROT135), (int(n_elements / 4), 1, 1))
        A = np.vstack((A0, A1, A2, A3))
        self.__embeddings = (A, B)

    def __reduce__(self):
        return (TriaGrid,
                (self.num_intervals, self.domain, self.identify_left_right, self.identify_bottom_top))

    def __str__(self):
        return (f'Tria-Grid on domain '
                f'[{self.x0_range[0]},{self.x0_range[1]}] x [{self.x1_range[0]},{self.x1_range[1]}]\n'
                f'x0-intervals: {self.x0_num_intervals}, x1-intervals: {self.x1_num_intervals}\n'
                f'elements: {self.size(0)}, edges: {self.size(1)}, vertices: {self.size(2)}')

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
            return super().subentities(codim, subentity_codim)

    def embeddings(self, codim=0):
        if codim == 0:
            return self.__embeddings
        else:
            return super().embeddings(codim)

    def bounding_box(self):
        return np.array(self.domain)

    @cached
    def orthogonal_centers(self):
        embeddings = self.embeddings(0)
        ne4 = len(embeddings[0]) // 4
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
            |NumPy array| of the data to visualize. If `U.dim == 2 and len(U) > 1`, the
            data is visualized as a time series of plots. Alternatively, a tuple of
            |Numpy arrays| can be provided, in which case a subplot is created for
            each entry of the tuple. The lengths of all arrays have to agree.
        codim
            The codimension of the entities the data in `U` is attached to (either 0 or 2).
        kwargs
            See :func:`~pymor.discretizers.builtin.gui.visualizers.PatchVisualizer.visualize`
        """
        from pymor.discretizers.builtin.gui.visualizers import PatchVisualizer
        from pymor.vectorarrays.interface import VectorArray
        from pymor.vectorarrays.numpy import NumpyVectorSpace
        if isinstance(U, (np.ndarray, VectorArray)):
            U = (U,)
        assert all(isinstance(u, (np.ndarray, VectorArray)) for u in U)
        U = tuple(NumpyVectorSpace.make_array(u) if isinstance(u, np.ndarray) else
                  u if isinstance(u.space, NumpyVectorSpace) else
                  NumpyVectorSpace.make_array(u.to_numpy())
                  for u in U)
        PatchVisualizer(self, codim=codim).visualize(U, **kwargs)

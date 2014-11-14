# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.grids.interfaces import AffineGridInterface
from pymor.grids.referenceelements import triangle


class TriaGrid(AffineGridInterface):
    """Basic implementation of a triangular grid on a rectangular domain.

    The global face, edge and vertex indices are given as follows ::

                 6---10----7---11----8
                 | \     6 | \     7 |
                 3   14    4   15    5
                 | 2     \ | 3     \ |
                 3----8----4----9----5
                 | \     4 | \     5 |
                 0   12    1   13    2
                 | 0    \  | 1     \ |
                 0----6----1----7----2

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
        n_elements = x0_num_intervals * x1_num_intervals * 2

        # TOPOLOGY
        self.__sizes = (n_elements,
                        ((x0_num_intervals + 1 - identify_left_right) * x1_num_intervals +
                         (x1_num_intervals + 1 - identify_bottom_top) * x0_num_intervals +
                         int(n_elements / 2)),
                        (x0_num_intervals + 1 - identify_left_right) * (x1_num_intervals + 1 - identify_bottom_top))

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

        D_EDGE_INDICES = np.arange(x0_num_intervals * x1_num_intervals, dtype=np.int32) + (num_v_edges + num_h_edges)

        E0 = np.array([D_EDGE_INDICES,
                       V_EDGE_INDICES[:, :-1].ravel(),
                       H_EDGE_INDICES[:-1, :].ravel()]).T
        E1 = np.array([D_EDGE_INDICES,
                       V_EDGE_INDICES[:, 1:].ravel(),
                       H_EDGE_INDICES[1:, :].ravel()]).T

        codim1_subentities = np.vstack((E0, E1))

        # calculate subentities -- codim-2
        VERTEX_H_INDICES = np.arange(x0_num_intervals + 1, dtype=np.int32)
        if identify_left_right:
            VERTEX_H_INDICES[-1] = 0
        VERTEX_V_INDICES = np.arange(x1_num_intervals + 1, dtype=np.int32)
        if identify_bottom_top:
            VERTEX_V_INDICES[-1] = 0
        VERTEX_V_INDICES *= x0_num_intervals + 1 - identify_left_right
        VERTEX_NUMERS = VERTEX_V_INDICES[:, np.newaxis] + VERTEX_H_INDICES

        V0 = np.array([VERTEX_NUMERS[:-1, :-1].ravel(),
                       VERTEX_NUMERS[:-1, 1:].ravel(),
                       VERTEX_NUMERS[1:, :-1].ravel()]).T
        V1 = np.array([VERTEX_NUMERS[1:, 1:].ravel(),
                       VERTEX_NUMERS[1:, :-1].ravel(),
                       VERTEX_NUMERS[:-1, 1:].ravel()]).T

        codim2_subentities = np.vstack((V0, V1))
        self.__subentities = (codim1_subentities, codim2_subentities)

        # GEOMETRY

        # embeddings
        x0_shifts0 = np.arange(x0_num_intervals) * self.x0_diameter + self.x0_range[0]
        x1_shifts0 = np.arange(x1_num_intervals) * self.x1_diameter + self.x1_range[0]
        x0_shifts1 = x0_shifts0 + self.x0_diameter
        x1_shifts1 = x1_shifts0 + self.x1_diameter
        B = np.vstack((np.array(np.meshgrid(x0_shifts0, x1_shifts0)).reshape((2, -1)).T,
                       np.array(np.meshgrid(x0_shifts1, x1_shifts1)).reshape((2, -1)).T))
        A0 = np.tile(np.diag([self.x0_diameter, self.x1_diameter]), (n_elements / 2, 1, 1))
        A1 = - A0
        A = np.vstack((A0, A1))
        self.__embeddings = (A, B)

    def __reduce__(self):
        return (TriaGrid,
                (self.num_intervals, self.domain, self.identify_left_right, self.identify_bottom_top))

    def __str__(self):
        return (('Tria-Grid on domain [{xmin},{xmax}] x [{ymin},{ymax}]\n' +
                 'x0-intervals: {x0ni}, x1-intervals: {x1ni}\n' +
                 'faces: {faces}, edges: {edges}, vertices: {vertices}')
                .format(xmin=self.x0_range[0], xmax=self.x0_range[1],
                        ymin=self.x1_range[0], ymax=self.x1_range[1],
                        x0ni=self.x0_num_intervals, x1ni=self.x1_num_intervals,
                        faces=self.size(0), edges=self.size(1), vertices=self.size(2)))

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

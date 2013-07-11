# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.exceptions import CodimError
from .interfaces import AffineGridInterface
from .referenceelements import triangle


class TriaGrid(AffineGridInterface):
    '''Ad-hoc implementation of a rectangular grid.

    The global face, edge and vertex indices are given as follows

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
        Tuple (n0, n1) determining a grid with n0 x n1 codim-0 entities.
    domain
        Tuple (ll, ur) where ll defines the lower left and ur the upper right
        corner of the domain.
    '''

    dim = 2
    dim_outer = 2
    reference_element = triangle

    def __init__(self, num_intervals=(2, 2), domain=[[0, 0], [1, 1]]):
        super(TriaGrid, self).__init__()
        self.num_intervals = num_intervals
        self.domain = np.array(domain)

        self.x0_num_intervals = num_intervals[0]
        self.x1_num_intervals = num_intervals[1]
        self.x0_range = self.domain[:, 0]
        self.x1_range = self.domain[:, 1]
        self.x0_width = self.x0_range[1] - self.x0_range[0]
        self.x1_width = self.x1_range[1] - self.x1_range[0]
        self.x0_diameter = self.x0_width / self.x0_num_intervals
        self.x1_diameter = self.x1_width / self.x1_num_intervals
        n_elements = self.x0_num_intervals * self.x1_num_intervals * 2

        # TOPOLOGY
        self.__sizes = (n_elements,
                        ((self.x0_num_intervals + 1) * self.x1_num_intervals +
                         (self.x1_num_intervals + 1) * self.x0_num_intervals +
                         int(n_elements / 2)),
                        (self.x0_num_intervals + 1) * (self.x1_num_intervals + 1))

        # calculate subentities -- codim-0
        edge_hoffset = (self.x0_num_intervals + 1) * self.x1_num_intervals
        edge_doffset = edge_hoffset + self.x0_num_intervals * (self.x1_num_intervals + 1)
        E0V = ((np.arange(self.x1_num_intervals, dtype=np.int32) * (self.x0_num_intervals + 1))[:, np.newaxis] +
               np.arange(self.x0_num_intervals, dtype=np.int32)).ravel()
        E0H = np.arange(n_elements / 2, dtype=np.int32) + edge_hoffset
        E0D = np.arange(n_elements / 2, dtype=np.int32) + edge_doffset

        E1V = E0V + 1
        E1H = E0H + self.x0_num_intervals
        E1D = E0D

        codim0_subentities = np.vstack((np.vstack((E0D, E0V, E0H)).T, np.vstack((E1D, E1V, E1H)).T))

        # calculate subentities -- codim-1

        V0 = E0V[:, np.newaxis] + np.array([0, 1, self.x0_num_intervals + 1], dtype=np.int32)
        V1 = E0V[:, np.newaxis] + np.array([self.x0_num_intervals + 2, self.x0_num_intervals + 1, 1], np.int32)
        codim1_subentities = np.vstack((V0, V1))
        self.__subentities = (codim0_subentities, codim1_subentities)

        # GEOMETRY

        # embeddings
        x0_shifts0 = np.arange(self.x0_num_intervals) * self.x0_diameter + self.x0_range[0]
        x1_shifts0 = np.arange(self.x1_num_intervals) * self.x1_diameter + self.x1_range[0]
        x0_shifts1 = x0_shifts0 + self.x0_diameter
        x1_shifts1 = x1_shifts0 + self.x1_diameter
        B = np.vstack((np.array(np.meshgrid(x0_shifts0, x1_shifts0)).reshape((2, -1)).T,
                       np.array(np.meshgrid(x0_shifts1, x1_shifts1)).reshape((2, -1)).T))
        A0 = np.tile(np.diag([self.x0_diameter, self.x1_diameter]), (n_elements / 2, 1, 1))
        A1 = - A0
        A = np.vstack((A0, A1))
        self.__embeddings = (A, B)
        self.lock()

    def __str__(self):
        return ('Tria-Grid on domain [{xmin},{xmax}] x [{ymin},{ymax}]\n' +
                'x0-intervals: {x0ni}, x1-intervals: {x1ni}\n' +
                'faces: {faces}, edges: {edges}, verticies: {verticies}').format(
                    xmin=self.x0_range[0], xmax=self.x0_range[1],
                    ymin=self.x1_range[0], ymax=self.x1_range[1],
                    x0ni=self.x0_num_intervals, x1ni=self.x1_num_intervals,
                    faces=self.size(0), edges=self.size(1), verticies=self.size(2))

    def size(self, codim=0):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        return self.__sizes[codim]

    def subentities(self, codim=0, subentity_codim=None):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        if subentity_codim is None:
            subentity_codim = codim + 1
        assert codim <= subentity_codim <= 2, CodimError('Invalid subentity codimension')
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

    @staticmethod
    def test_instances():
        '''Used for unit testing.'''
        return [TriaGrid((2, 4)), TriaGrid((1, 1)), TriaGrid((42, 42))]

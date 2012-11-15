from __future__ import absolute_import, division, print_function, unicode_literals

import math as m
import numpy as np

from .exceptions import CodimError
from .base import Base


class Rect(Base):
    '''Ad-hoc implementation of a rectangular grid.

    The global face, edge and vertex indices are given as follows

                 6--10---7--11---8
                 |       |       |
                 3   2   4   3   5
                 |       |       |
                 3---8---4---9---5
                 |       |       |
                 0   0   1   1   2
                 |       |       |
                 0---6---1---7---2

    '''

    dim = 2

    def __init__(self, num_intervals=(2, 2), domain=[[0, 0], [1, 1]]):
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
        self.diameter_max = max(self.x0_diameter, self.x1_diameter)
        self.diameter_min = min(self.x0_diameter, self.x1_diameter)
        n_elements = self.x0_num_intervals * self.x1_num_intervals

        # TOPOLOGY

        # calculate subentities -- codim-0
        EVL = ((np.arange(self.x1_num_intervals) * (self.x0_num_intervals + 1))[:, np.newaxis] +
               np.arange(self.x0_num_intervals)).ravel()
        EVR = ((np.arange(self.x1_num_intervals) * (self.x0_num_intervals + 1))[:, np.newaxis] +
               np.arange(1, self.x0_num_intervals + 1)).ravel()
        EHB = np.arange(n_elements) + (self.x0_num_intervals + 1) * self.x1_num_intervals
        EHT = np.arange(n_elements) + (self.x0_num_intervals + 1) * self.x1_num_intervals + self.x0_num_intervals
        codim0_subentities = np.array((EHB, EVR, EHT, EVL), dtype=np.int32)

        # calculate subentities -- codim-1
        VVB = np.arange((self.x0_num_intervals + 1) * self.x1_num_intervals)
        VVT = np.arange((self.x0_num_intervals + 1) * self.x1_num_intervals) + (self.x0_num_intervals + 1)
        VHL = ((np.arange(self.x1_num_intervals + 1) * (self.x0_num_intervals + 1))[:, np.newaxis] +
               np.arange(self.x0_num_intervals)).ravel()
        VHR = VHL + 1
        codim1_subentities = np.array(np.hstack((np.vstack((VVB, VVT)), np.vstack((VHL, VHR)))), dtype=np.int32)

        self._subentities = (codim0_subentities, codim1_subentities)

        # GEOMETRY

        # calculate centers -- codim-0
        x0_centers = np.arange(self.x0_num_intervals) * self.x0_diameter + self.x0_diameter / 2 + self.x0_range[0]
        x1_centers = np.arange(self.x1_num_intervals) * self.x1_diameter + self.x1_diameter / 2 + self.x1_range[0]
        codim0_centers = np.array(np.meshgrid(x0_centers, x1_centers)).reshape((2, -1))

        # calculate centers -- codim-1
        x0_centers_2 = np.arange(self.x0_num_intervals + 1) * self.x0_diameter + self.x0_range[0]
        x1_centers_2 = np.arange(self.x1_num_intervals + 1) * self.x1_diameter + self.x1_range[0]
        codim1_centers = np.hstack((np.array(np.meshgrid(x0_centers_2, x1_centers)).reshape((2, -1)),
                                    np.array(np.meshgrid(x0_centers, x1_centers_2)).reshape((2, -1))))

        # calculate centers -- codim-2
        codim2_centers = np.array(np.meshgrid(x0_centers_2, x1_centers_2)).reshape((2, -1))

        self._centers = (codim0_centers, codim1_centers, codim2_centers)

        # diameters / volumes
        codim0_diameters = np.empty_like(codim0_centers[0])
        codim0_diameters.fill(m.sqrt(self.x0_diameter ** 2 + self.x1_diameter ** 2))
        codim0_volumes = np.empty_like(codim0_diameters)
        codim0_volumes.fill(self.x0_diameter * self.x1_diameter)

        codim1_diameters = np.empty_like(codim1_centers[0])
        codim1_diameters[:(self.x0_num_intervals + 1) * self.x1_num_intervals] = self.x1_diameter
        codim1_diameters[(self.x0_num_intervals + 1) * self.x1_num_intervals:] = self.x0_diameter
        codim1_volumes = codim1_diameters

        self._volumes = (codim0_volumes, codim1_volumes)
        self._volumes_inverse = (np.reciprocal(codim0_volumes), np.reciprocal(codim1_volumes))
        self._diameters = (codim0_diameters, codim1_diameters)

        # outer normals
        self._normals = np.empty((2, 4, n_elements))
        self._normals[0, 0, :] = 0
        self._normals[1, 0, :] = -1
        self._normals[0, 1, :] = 1
        self._normals[1, 1, :] = 0
        self._normals[0, 2, :] = 0
        self._normals[1, 2, :] = 1
        self._normals[0, 3, :] = -1
        self._normals[1, 3, :] = 0

    def __str__(self):
        return ('Rect-Grid on domain [{xmin},{xmax}] x [{ymin},{ymax}]\n' +
                'x0-intervals: {x0ni}, x1-intervals: {x1ni}\n' +
                'faces: {faces}, edges: {edges}, verticies: {verticies}').format(
                    xmin=self.x0_range[0], xmax=self.x0_range[1],
                    ymin=self.x1_range[0], ymax=self.x1_range[1],
                    x0ni=self.x0_num_intervals, x1ni=self.x1_num_intervals,
                    faces=self.size(0), edges=self.size(1), verticies=self.size(2))

    def subentities(self, codim=0, subentity_codim=None):
        assert 0 <= codim <= 1, CodimError('Invalid codimension')
        if subentity_codim is None or subentity_codim == codim + 1:
            return self._subentities[codim]
        else:
            return super(Rect, self).subentities(codim, subentity_codim)

    def size(self, codim=0):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        return self._centers[codim].shape[1]

    def centers(self, codim=0):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        return self._centers[codim]

    def volumes(self, codim=0):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        if codim == 2:
            return np.ones(self.size(2))
        else:
            return self._volumes[codim]

    def volumes_inverse(self, codim=0):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        if codim == 2:
            return np.ones(self.size(2))
        else:
            return self._volumes_inverse[codim]

    def diameters(self, codim=0):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        if codim == 2:
            return np.ones(self.size(2))
        else:
            return self._diameters[codim]

    def unit_outer_normals(self):
        '''only at centers?'''
        return self._normals


    #def center_distances(self):
     #   raise NotImplementedError

    #def jacobian_inverse_transposed(self):
     #   raise NotImplementedError

    #def quadrature_points(self, codim=0, order=0):
       # assert 0 <= codim <= 1, CodimError('Invalid codimension')
       # raise NotImplementedError

    #def quadrature_weights(self, codim=0, order=0):
       # assert 0 <= codim <= 1, CodimError('Invalid codimension')
       # raise NotImplementedError

    #def S(self):
        #'''to be renamed'''
        #pass

    #def ES(self):
        #'''to be renamed'''
        #pass

    #def DS(self):
        #'''to be renamed'''
        #pass

    #def alpha(self):
        #'''to be renamed'''
        #pass

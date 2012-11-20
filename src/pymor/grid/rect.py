from __future__ import absolute_import, division, print_function, unicode_literals

import math as m
import numpy as np

from pymor.core.exceptions import CodimError
from .interfaces import ISimpleAffineGrid
from .referenceelements import square


class Rect(ISimpleAffineGrid):
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
    dim_outer = 2

    def __init__(self, num_intervals=(2, 2), domain=[[0, 0], [1, 1]]):
        self.num_intervals = num_intervals
        self.domain = np.array(domain)

        self.reference_element = square
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
        self._sizes = (n_elements,
                       (  (self.x0_num_intervals + 1) * self.x1_num_intervals +
                          (self.x1_num_intervals + 1) * self.x0_num_intervals   ),
                       (self.x0_num_intervals + 1) * (self.x1_num_intervals + 1) )

        # calculate subentities -- codim-0
        EVL = ((np.arange(self.x1_num_intervals, dtype=np.int32) * (self.x0_num_intervals + 1))[:, np.newaxis] +
               np.arange(self.x0_num_intervals, dtype=np.int32)).ravel()
        EVR = EVL + 1
        EHB = np.arange(n_elements, dtype=np.int32) + (self.x0_num_intervals + 1) * self.x1_num_intervals
        EHT = EHB + self.x0_num_intervals
        codim0_subentities = np.array((EHB, EVR, EHT, EVL)).T

        # calculate subentities -- codim-1
        codim1_subentities = (np.tile(EVL[:, np.newaxis], 4) +
                              np.array([0, 1, self.x0_num_intervals + 2, self.x0_num_intervals + 1], dtype=np.int32))
        self._subentities = (codim0_subentities, codim1_subentities)


        # GEOMETRY

        # embeddings
        x0_shifts = np.arange(self.x0_num_intervals) * self.x0_diameter + self.x0_range[0]
        x1_shifts = np.arange(self.x1_num_intervals) * self.x1_diameter + self.x1_range[0]
        shifts = np.array(np.meshgrid(x0_shifts, x1_shifts)).reshape((2, -1))
        A = np.tile(np.diag([self.x0_diameter, self.x1_diameter]), (n_elements, 1, 1))
        B = shifts.T
        self._embeddings = (A, B)


    def __str__(self):
        return ('Rect-Grid on domain [{xmin},{xmax}] x [{ymin},{ymax}]\n' +
                'x0-intervals: {x0ni}, x1-intervals: {x1ni}\n' +
                'faces: {faces}, edges: {edges}, verticies: {verticies}').format(
                    xmin=self.x0_range[0], xmax=self.x0_range[1],
                    ymin=self.x1_range[0], ymax=self.x1_range[1],
                    x0ni=self.x0_num_intervals, x1ni=self.x1_num_intervals,
                    faces=self.size(0), edges=self.size(1), verticies=self.size(2))

    def size(self, codim=0):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        return self._sizes[codim]

    def subentities(self, codim=0, subentity_codim=None):
        assert 0 <= codim <= 1, CodimError('Invalid codimension')
        if subentity_codim is None:
            subentity_codim = codim + 1
        if codim == 0:
            return self._subentities[subentity_codim - 1]
        else:
            return super(Rect, self).subentities(codim, subentity_codim)

    def embeddings(self, codim=0):
        if codim == 0:
            return self._embeddings
        else:
            return super(Rect, self).embeddings(codim)

    def visualize(self, dofs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        assert dofs.size == self.size(), ValueError('DOF-vector has the wrong size')
        im = plt.imshow(dofs.reshape((self.x1_num_intervals, self.x0_num_intervals)), cmap=cm.jet,
                        aspect=self.x1_diameter / self.x0_diameter, extent=self.domain.T.ravel(),
                        interpolation='none')

        # make sure, the colorbar has the right height: (from mpl documentation)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(im, cax=cax)
        plt.show()

    #def center_distances(self):
     #   raise NotImplementedError


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

if __name__ == '__main__':
    g = Rect(num_intervals=(120,60), domain=[[0,0],[2,1]])
    X = np.sin(2*np.pi*g.centers()[0,:])*np.sin(2*np.pi*g.centers()[1,:])
    g.visualize(X)

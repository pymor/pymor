# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.exceptions import CodimError
from pymor.grids.interfaces import AffineGridInterface
from pymor.grids.referenceelements import square


class RectGrid(AffineGridInterface):
    '''Ad-hoc implementation of a rectangular grid.

    The global face, edge and vertex indices are given as follows

                 x1
                 ^
                 |

                 6--10---7--11---8
                 |       |       |
                 3   2   4   3   5
                 |       |       |
                 3---8---4---9---5
                 |       |       |
                 0   0   1   1   2
                 |       |       |
                 0---6---1---7---2  --> x0

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
    reference_element = square

    def __init__(self, num_intervals=(2, 2), domain=[[0, 0], [1, 1]],
                 identify_left_right=False, identify_bottom_top=False):
        if identify_left_right:
            assert num_intervals[0] > 1
        if identify_bottom_top:
            assert num_intervals[1] > 1
        super(RectGrid, self).__init__()
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
        ni0, ni1 = num_intervals

        # TOPOLOGY

        # mapping of structured indices to global indices
        structured_to_global_0 = np.arange(ni0 * ni1, dtype=np.int32).reshape((ni1, ni0)).swapaxes(0, 1)
        structured_to_global_2 = (np.arange((ni0 + 1) * (ni1 + 1), dtype=np.int32)
                                  .reshape(((ni1 + 1), (ni0 + 1))).swapaxes(0, 1))
        self._structured_to_global = [structured_to_global_0, None, structured_to_global_2]
        global_to_structured_0 = np.empty((ni0 * ni1, 2), dtype=np.int32)
        global_to_structured_0[:, 0] = np.tile(np.arange(ni0, dtype=np.int32), ni1)
        global_to_structured_0[:, 1] = np.repeat(np.arange(ni1, dtype=np.int32), ni0)
        global_to_structured_1 = np.empty(((ni0 + 1) * (ni1 + 1), 2), dtype=np.int32)
        global_to_structured_1[:, 0] = np.tile(np.arange(ni0 + 1, dtype=np.int32), ni1 + 1)
        global_to_structured_1[:, 1] = np.repeat(np.arange(ni1 + 1, dtype=np.int32), ni0 + 1)
        self._global_to_structured = [global_to_structured_0, None, global_to_structured_1]

        # calculate subentities -- codim-0
        codim1_subentities = np.empty((ni1, ni0, 4), dtype=np.int32)
        if identify_left_right:
            codim1_subentities[:, :,  3] = np.arange(ni0 * ni1).reshape((ni1, ni0))
            codim1_subentities[:, :,  1] = codim1_subentities[:, :, 3] + 1
            codim1_subentities[:, -1, 1] = codim1_subentities[:, 0, 3]
        else:
            codim1_subentities[:, :,  3] = (np.arange(ni0)[np.newaxis, :] + (np.arange(ni1) * (ni0 + 1))[:, np.newaxis])
            codim1_subentities[:, :,  1] = codim1_subentities[:, :, 3] + 1
        offset = np.max(codim1_subentities[:, :, [1, 3]]) + 1
        codim1_subentities[:, :, 0] = (np.arange(ni0 * ni1) + offset).reshape((ni1, ni0))
        codim1_subentities[:, :, 2] = codim1_subentities[:, :, 0] + num_intervals[0]
        if identify_bottom_top:
            codim1_subentities[-1, :, 2] = codim1_subentities[0, :, 0]
        codim1_subentities = codim1_subentities.reshape((-1, 4))

        # calculate subentities -- codim-1
        codim2_subentities = np.empty((ni1, ni0, 4), dtype=np.int32)
        if identify_left_right:
            codim2_subentities[:, :,  0] = np.arange(ni0 * ni1).reshape((ni1, ni0))
            codim2_subentities[:, :,  1] = codim2_subentities[:, :, 0] + 1
            codim2_subentities[:, -1, 1] = codim2_subentities[:, 0, 0]
            codim2_subentities[:, :,  3] = codim2_subentities[:, :, 0] + ni0
            codim2_subentities[:, :,  2] = codim2_subentities[:, :, 1] + ni0
        else:
            codim2_subentities[:, :,  0] = (np.arange(ni0)[np.newaxis, :] +
                                            (np.arange(ni1) * (ni0 + 1))[:, np.newaxis])
            codim2_subentities[:, :,  1] = codim2_subentities[:, :, 0] + 1
            codim2_subentities[:, :,  3] = codim2_subentities[:, :, 0] + (ni0 + 1)
            codim2_subentities[:, :,  2] = codim2_subentities[:, :, 0] + (ni0 + 2)
        if identify_bottom_top:
            codim2_subentities[-1, :, 3] = codim2_subentities[0, :, 0]
            codim2_subentities[-1, :, 2] = codim2_subentities[0, :, 1]
        codim2_subentities = codim2_subentities.reshape((-1, 4))

        self.__subentities = (codim1_subentities, codim2_subentities)

        # calculate number of elements
        self.__sizes = (n_elements,
                        np.max(codim1_subentities) + 1,
                        np.max(codim2_subentities) + 1)

        # GEOMETRY

        # embeddings
        x0_shifts = np.arange(self.x0_num_intervals) * self.x0_diameter + self.x0_range[0]
        x1_shifts = np.arange(self.x1_num_intervals) * self.x1_diameter + self.x1_range[0]
        shifts = np.array(np.meshgrid(x0_shifts, x1_shifts)).reshape((2, -1))
        A = np.tile(np.diag([self.x0_diameter, self.x1_diameter]), (n_elements, 1, 1))
        B = shifts.T
        self.__embeddings = (A, B)
        self.lock()

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
        return self.__sizes[codim]

    def subentities(self, codim=0, subentity_codim=None):
        assert 0 <= codim <= 2, CodimError('Invalid codimension')
        if subentity_codim is None:
            subentity_codim = codim + 1
        assert codim <= subentity_codim <= self.dim, CodimError('Invalid subentity codimensoin')
        if codim == 0:
            if subentity_codim == 0:
                return np.arange(self.size(0), dtype='int32')[:, np.newaxis]
            else:
                return self.__subentities[subentity_codim - 1]
        else:
            return super(RectGrid, self).subentities(codim, subentity_codim)

    def embeddings(self, codim=0):
        if codim == 0:
            return self.__embeddings
        else:
            return super(RectGrid, self).embeddings(codim)

    def structured_to_global(self, codim):
        '''Returns an array which maps structured indices to global codim-`codim` indices.

        In other words `structed_to_global(codim)[i, j]` is the global index of the i-th in
        x0-direction and j-th in x1-direction codim-`codim` entity of the grid.
        '''
        if codim not in (0, 2):
            raise NotImplementedError
        return self._structured_to_global[codim]

    def global_to_structured(self, codim):
        '''Returns an array which maps global codim-`codim` indices to structured indices.

        I.e. if `GTS = global_to_structured(codim)` and `STG = structured_to_global(codim)`, then
        `STG[GTS[:, 0], GTS[:, 1]] == numpy.arange(size(codim))`.
        '''
        if codim not in (0, 2):
            raise NotImplementedError
        return self._global_to_structured[codim]

    def vertex_coordinates(self, dim):
        '''Returns an array of the x_dim koordinates of the grid verticies. I.e. ::

            centers(2)[structured_to_global(2)[i, j]] == np.array([vertex_coordinates(0)[i], vertex_coordinates(1)[j]])
        '''
        assert 0 <= dim < 2
        return np.linspace(self.domain[0, dim], self.domain[1, dim], self.num_intervals[dim] + 1)

    def visualize(self, dofs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        assert dofs.size == self.size(0), ValueError('DOF-vector has the wrong size')
        im = plt.imshow(dofs.reshape((self.x1_num_intervals, self.x0_num_intervals)), cmap=cm.jet,
                        aspect=self.x1_diameter / self.x0_diameter, extent=self.domain.T.ravel(),
                        interpolation='none')

        # make sure, the colorbar has the right height: (from mpl documentation)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(im, cax=cax)
        plt.show()

    @staticmethod
    def test_instances():
        '''Used for unit testing.'''
        return [RectGrid((2, 4)),  RectGrid((1, 1)), RectGrid((42, 42)),
                RectGrid((2, 4), identify_left_right=True),
                RectGrid((2, 4), identify_bottom_top=True),
                RectGrid((2, 4), identify_left_right=True, identify_bottom_top=True),
                RectGrid((2, 1), identify_left_right=True),
                RectGrid((1, 2), identify_bottom_top=True),
                RectGrid((2, 2), identify_left_right=True, identify_bottom_top=True)]

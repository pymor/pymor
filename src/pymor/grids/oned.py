#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import print_function
from __future__ import division
import numpy as np

from pymor.core.exceptions import CodimError
from pymor.grids.interfaces import AffineGridInterface
from .referenceelements import line


class OnedGrid(AffineGridInterface):
    '''One-dimensional grid on an interval.

    Parameters
    ----------
    domain
        Tuple (left, right) containing the left and right boundary of the domain.
    num_intervals
        The number of codim-0 entities.
    '''

    dim = 1
    dim_outer = 1
    reference_element = line

    def __init__(self, domain=(0, 1), num_intervals=4):
        super(OnedGrid, self).__init__()
        self.reference_element = line
        self._domain = np.array(domain)
        self._num_intervals = num_intervals
        self._width = np.abs(self._domain[1] - self._domain[0]) / self._num_intervals
        self.__subentities = np.vstack((np.arange(self._num_intervals, dtype=np.int32),
                                        np.arange(self._num_intervals, dtype=np.int32) + 1))
        self.__A = np.ones(self._num_intervals, dtype=np.int32)[:, np.newaxis, np.newaxis] * self._width
        self.__B = (self._domain[0] + self._width * (np.arange(self._num_intervals, dtype=np.int32)))[:, np.newaxis]
        self.lock()

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
        if 0 <= codim <= 1:
            return self._num_intervals + codim

    def subentities(self, codim=0, subentity_codim=None):
        assert 0 <= codim <= 1, CodimError('Invalid codimension')
        if subentity_codim is None:
            subentity_codim = codim + 1
        assert codim <= subentity_codim <= self.dim, CodimError('Invalid subentity codimensoin')
        if codim == 0:
            if subentity_codim == 0:
                return np.arange(self.size(0), dtype='int32')[:, np.newaxis]
            else:
                return self.__subentities.T
        else:
            return super(OnedGrid, self).subentities(codim, subentity_codim)

    def embeddings(self, codim=0):
        if codim == 0:
            return self.__A, self.__B
        else:
            return super(OnedGrid, self).embeddings(codim)

    @staticmethod
    def test_instances():
        '''Used for unit testing.'''
        return [OnedGrid(domain=np.array((-2, 2)), num_intervals=10),
                OnedGrid(domain=np.array((-2, -4)), num_intervals=100),
                OnedGrid(domain=np.array((3, 2)), num_intervals=10),
                OnedGrid(domain=np.array((1, 2)), num_intervals=10000)]

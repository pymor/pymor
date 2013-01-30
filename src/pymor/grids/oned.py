#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import numpy as np

from pymor.core.exceptions import CodimError
from pymor.grids.interfaces import AffineGridInterface
from .referenceelements import line


class OnedGrid(AffineGridInterface):

    dim = 1
    dim_outer = 1
    reference_element = line
    id = 'grid.oned'

    def __init__(self, interval=np.array((0, 1)), size=4):
        self.reference_element = line
        self._interval = interval
        self._size = size
        self._width = np.abs(self._interval[1] - self._interval[0]) / self._size
        self.__subentities = np.vstack((np.arange(self._size, dtype=np.int32),
                                        np.arange(self._size, dtype=np.int32) + 1))
        self.__A = np.ones(self._size, dtype=np.int32)[:, np.newaxis, np.newaxis] * self._width
        self.__B = (self._interval[0] + self._width * (np.arange(self._size, dtype=np.int32)))[:, np.newaxis]

    def __str__(self):
        return (self.id + ', domain [{xmin},{xmax}]'
                + ', {elements} elements'
                + ', {vertices} vertices'
                ).format(xmin=self._interval[0],
                         xmax=self._interval[1],
                         elements=self.size(0),
                         vertices=self.size(1))

    def size(self, codim=0):
        assert 0 <= codim <= 1, 'codim has to be between 0 and {}!'.format(self.dim)
        if 0 <= codim <= 1:
            return self._size + codim

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

    def test_instances():
        return [OnedGrid(interval=np.array((-2, 2)), size=10),
                OnedGrid(interval=np.array((-2, -4)), size=100),
                OnedGrid(interval=np.array((3, 2)), size=10),
                OnedGrid(interval=np.array((1, 2)), size=10000)]

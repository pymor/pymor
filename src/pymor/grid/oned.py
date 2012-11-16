#!/usr/bin/env python

from __future__ import division
import numpy as np

from .exceptions import CodimError
from .interfaces import IGrid


class Oned(IGrid):

    dim = 1

    def __init__(self, interval=np.array((0, 1)), size=4):
        self._interval = interval
        self._size = size
        self._width = np.abs(self._interval[1] - self._interval[0]) / self._size
        self._centers = self._interval[0] + self._width * (np.arange(self._size) + 0.5)
        self._vertices = self._interval[0] + self._width * np.arange(self._size + 1)
        self._volumes_codim_0 = self._width * np.ones((1, self._size))
        self._volumes_codim_1 = np.ones((1, self._size))
        self._subentities = np.vstack((np.arange(self._size), np.arange(self._size) + 1))

    def name(self):
        return 'grid.oned'

    def size(self, codim=0):
        if codim == 0:
            return self._size
        elif codim == 1:
            return self._size + 1
        else:
            raise ValueError('in pymor.' + self.name() + ': codim has to be between 0 and ' + self.dim + '!')

    def subentities(self, codim=0, subentity_codim=None):
        assert codim == 0, CodimError('Invalid codimension')
        assert subentity_codim is None or subentity_codim == 1, CodimError('Invalid subentity codimension')
        return self._subentities

    def centers(self, codim=0):
        '''
        Since vectors in pymor are column vectors,
        centers(codim) returns a matrix of size (dim - codim) x size(codim)
        '''
        if codim == 0:
            return self._centers
        elif codim == 1:
            return self._vertices
        else:
            raise ValueError('in pymor.' + self.name() + ': codim has to be between 0 and ' + self.dim + '!')

    def volumes(self, codim=0):
        if codim == 0:
            return self._volumes_codim_0
        elif codim == 1:
            return self._volumes_codim_1
        else:
            raise ValueError('in pymor.' + self.name() + ': codim has to be between 0 and ' + self.dim + '!')


if __name__ == '__main__':
    g = Oned()
    print(g.centers(0))
    print(g.centers(1))
    print(g.volumes(0))
    print(g.volumes(1))

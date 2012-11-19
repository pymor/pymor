#!/usr/bin/env python

from __future__ import print_function
from pymor.core import interfaces


class Interface(interfaces.BasicInterface):

    id = 'common.discreteoperator.stationary.linear'
    size = (0, 0)

    def apply(self):
        pass

    def __str__(self):
        return '{id}, size {rows}x{cols}'.format(id=self.id,
                                                 rows=self.size[0],
                                                 cols=self.size[1])


class ScipySparse(Interface):

    id = Interface.id + '.scipysparse'

    def __init__(self, matrix):
        '''
        here should be a contract to check if matrix is a scipy.sparse
        '''
        self.matrix = matrix
        self.size = self.matrix.shape

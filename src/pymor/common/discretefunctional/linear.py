#!/usr/bin/env python

# pymor
from pymor.core import interfaces


class Interface(interfaces.BasicInterface):

    id = 'common.discretefunctional.linear'
    size = 0

    def __str__(self):
        return '{id}, size {size}'.format(id=self.id,
                                          size=self.size)


class NumpyDense(Interface):

    id = Interface.id + '.numpydense'

    def __init__(self, vector):
        '''
        here should be a contract to check if vector is a numpy.ndarray
        '''
        self.vector = vector
        self.size = self.vector.size

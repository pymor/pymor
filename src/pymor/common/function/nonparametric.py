#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from pymor import core


class Interface(core.BasicInterface):

    id = 'common.function.nonparametric'
    dim_domain = 0
    dim_range = 0

    @core.interfaces.abstractmethod
    def evaluate(self, x):
        raise NotImplementedError()


class Constant(Interface):

    id = Interface.id + '.constant'
    dim_domain = 1
    dim_range = 1
    name = id

    def __init__(self, value=1.0, dim_domain=1, dim_range=1, name=id):
        '''
        here should be a contract to enforce that np.array(value) is valid
        '''
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        self.name = name
        value = np.array(value, copy=False)
        if value.size == self.dim_range:
            self._value = value.reshape(self.dim_range, 1)
        else:
            raise ValueError('Given value has wrong size!')

    def evaluate(self, x):
        '''
        here should be a contract to enforce that np.array(x) is valid
        '''
        x = np.array(x, copy=False, ndmin=1)
        if x.ndim == 1:
            if x.size == self.dim_domain:
                return self._value
            else:
                raise ValueError('Given value has wrong size!')
        elif x.ndim == 2:
            if x.shape[0] == self.dim_domain:
                return np.tile(value, (1, self.dim_range))
            else:
                raise ValueError('Given value has wrong size!')
        else:
            raise ValueError('Given value has wrong size!')


if __name__ == '__main__':
    print('testing ', end='')
    value = 3.7
    f = Constant(value)
    print(f.name + '... ', end='')
    result = f.evaluate(value)
    if result == np.array(value):
        print('done')
    else:
        print('failed')
        print(value)
        print(result)
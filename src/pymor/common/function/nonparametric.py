#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from pymor import core


class Interface(core.BasicInterface):

    name = 'common.function.nonparametric'

    dim_domain = 0

    dim_range = 0

    @core.interfaces.abstractmethod
    def evaluate(self, x):
        pass


class Constant(Interface):

    name = 'common.function.nonparametric.constant'

    dim_domain = 1

    dim_range = 1

    def __init__(self, value=1.0, dim_domain=1, dim_range=1, name='common.function.nonparametric.constant'):
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        if type(value) is not np.ndarray:
            value = np.array(value)
        if value.size == self.dim_range:
            self._value = value.reshape(self.dim_range, 1)
        else:
            raise ValueError('in pymor.' + self.name() + ': given value has wrong size!')
        self.name = name

    def evaluate(self, x):
        if type(x) is not np.ndarray:
            x = np.array(x)
        print(x.ndim)
        if x.ndim == 1:
            if x.size == self.dim_domain:
                return self._value
            else:
                raise ValueError('in pymor.' + self.name + ': given x has wrong size!')
        elif x.ndim == 2:
            if x.shape[0] == self.dim_domain:
                a
            else:
                raise ValueError('in pymor.' + self.name + ': given x has wrong size!')


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
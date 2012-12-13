#!/usr/bin/env python

# needed for name == main
from __future__ import print_function

# numpy
import numpy as np

# pymor
from pymor.core import interfaces


class Interface(interfaces.BasicInterface):

    id = 'common.function.nonparametric'
    dim_domain = 0
    dim_range = 0
    name = id

    def __str__(self):
        return ('{name} ({id}): R^{dim_domain} -> R^{dim_range}'
                ).format(name=self.name,
                         dim_domain=self.dim_domain,
                         dim_range=self.dim_range,
                         id=self.id)

    @interfaces.abstractmethod
    def evaluate(self, x):
        pass

    def __call__(self, x):
        return self.evaluate(x)


class Constant(Interface):

    id = Interface.id + '.constant'
    dim_domain = 1
    dim_range = 1
    name = id

    def __init__(self, value=1.0, dim_domain=1, dim_range=1, name=id):
        '''
        here should be a contract to enforce that np.array(value, copy=False) is valid
        '''
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        self.name = name
        value = np.array(value, copy=False)
        if value.size == self.dim_range:
            if self.dim_range == 1:
                self._value = value
            else:
                self._value = value.reshape(1, self.dim_range)
        else:
            raise ValueError('Given value has wrong size!')

    def __str__(self):
        return ('{base}, x -> {value}').format(base=Interface.__str__(self),
                                                 value=self._value)

    def evaluate(self, x):
        '''
        \todo    here should be a contract to enforce that np.array(x, copy=False, ndmin=1) is valid
        '''
        x = np.array(x, copy=False, ndmin=1)
        number_of_input_points = 0
        if self.dim_domain == 1:
            if x.ndim == 1:
                number_of_input_points = x.size
            elif x.ndim == 2:
                assert x.shape[0] == 1 or x.shape[1] == 1, 'Given x has wrong size!'
                number_of_input_points = x.size
        else:
            if x.ndim == 1:
                assert x.size == self.dim_domain, 'Given x has wrong size!'
                number_of_input_points = 1
            else:
                assert x.ndim == 2, 'Given x has wrong size!'
                assert x.shape[1] == self.dim_domain, 'Given x has wrong size!'
                number_of_input_points = x.shape[0]
        if self.dim_range == 1:
            return self._value * np.ones(number_of_input_points)
        else:
            return np.tile(self._value, (number_of_input_points, 1))


if __name__ == '__main__':
    def test_function(function, arguments):
        print('testing {function}: '.format(function=function))
        for argument in arguments:
            print('    {name}({argument}) = {result}'.format(name=function.name,
                                                             argument=argument,
                                                             result=function.evaluate(argument)))
    f_1_1 = Constant(dim_domain=1, dim_range=1, value=1., name='f_1_1')
    x_scalar = 1.
    x_scalar_array = np.array(x_scalar)
    x_list = [1., 1., 1.]
    x_list_array = np.array(x_list)
    x_ones_array = np.ones((1, 3))
    x_ones_array_transposed = np.ones((3, 1))
    xes = [x_scalar, x_scalar_array, x_list_array, x_ones_array, x_ones_array_transposed]
    test_function(f_1_1, xes)
    f_1_2 = Constant(dim_domain=1, dim_range=2, value=[0., 1.], name='f_1_2')
    test_function(f_1_2, xes)
    f_2_1 = Constant(dim_domain=2, dim_range=1, value=1., name='f_2_1')
    x_vector = [1., 1.]
    x_vector_array = np.array(x_vector)
    x_vectors_array = np.array([[1., 1.], [1., 1.], [1., 1.]])
    x_ones_vectors_array = np.ones((3, 2))
    xes = [x_vector, x_vector_array, x_vectors_array, x_ones_vectors_array]
    test_function(f_2_1, xes)
    f_2_2 = Constant(dim_domain=2, dim_range=2, value=[1., 2.], name='f_2_2')
    test_function(f_2_2, xes)

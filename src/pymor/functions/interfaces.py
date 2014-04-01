# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core import ImmutableInterface, abstractmethod
from pymor.parameters import Parametric
from pymor.tools import Named


class FunctionInterface(ImmutableInterface, Parametric, Named):
    '''Interface for |Parameter| dependent analytical functions.

    Every function is a map of the form ::

       f(μ): Ω ⊆ R^d -> R^(shape_range)

    The returned values are |NumPy arrays| of arbitrary (but fixed)
    shape. While the function could raise an error if it is evaluated
    for an argument not in Ω, the exact behavior is left undefined.

    Functions are vectorized in the sense, that if `x.ndim = k`, then ::

       f(x, μ)[i0, i1, ..., i(k-2)] = f(x[i0, i1, ..., i(k-2)], μ),

    in particular `f(x, μ).shape == x.shape[:-1] + shape_range`.

    Attributes
    ----------
    dim_domain
        The dimension d > 0.
    shape_range
        The shape of the function values.
    '''

    @abstractmethod
    def evaluate(self, x, mu=None):
        '''Evaluate the function for given argument and |Parameter|.'''
        pass

    def __call__(self, x, mu=None):
        '''Shorthand for :meth:`~FunctionInterface.evaluate`.'''
        return self.evaluate(x, mu)

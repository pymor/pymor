# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import ImmutableInterface, abstractmethod
from pymor.parameters.base import Parametric


class FunctionInterface(ImmutableInterface, Parametric):
    """Interface for |Parameter| dependent analytical functions.

    Every function is a map of the form ::

       f(μ): Ω ⊆ R^d -> R^(shape_range)

    The returned values are |NumPy arrays| of arbitrary (but fixed)
    shape. Note that NumPy distinguished between one-dimensional
    arrays of length 1 (with shape `(1,)`) and zero-dimensional
    scalar arrays (with shape `()`). In pyMOR, we usually
    expect scalar-valued functions to have `shape_range == ()`.

    While the function might raise an error if it is evaluated
    for an argument not in the domain Ω, the exact behavior is left
    undefined.

    Functions are vectorized in the sense, that if `x.ndim = k`, then ::

       f(x, μ)[i0, i1, ..., i(k-2)] = f(x[i0, i1, ..., i(k-2)], μ),

    in particular `f(x, μ).shape == x.shape[:-1] + shape_range`.

    Attributes
    ----------
    dim_domain
        The dimension d > 0.
    shape_range
        The shape of the function values.
    """

    @abstractmethod
    def evaluate(self, x, mu=None):
        """Evaluate the function for given argument and |Parameter|."""
        pass

    def __call__(self, x, mu=None):
        """Shorthand for :meth:`~FunctionInterface.evaluate`."""
        return self.evaluate(x, mu)

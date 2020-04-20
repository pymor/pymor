# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.defaults import defaults


@defaults('rtol', 'atol')
def float_cmp(x, y, rtol=1e-14, atol=1e-14):
    """Compare x and y component-wise for almost equality.

    For scalars we define almost equality as ::

       float_cmp(x,y) <=> |x - y| <= atol + |y|*rtol

    .. note::
       Numpy's :meth:`~numpy.allclose` method uses the same definition but
       treats arrays containing infinities as close if the infinities are
       at the same places and all other entries are close.
       In our definition, arrays containing infinities can never be close
       which seems more appropriate in most cases.

    Parameters
    ----------
    x, y
        |NumPy arrays| to be compared. Have to be broadcastable to the same shape.
    rtol
        The relative tolerance.
    atol
        The absolute tolerance.
    """

    return np.abs(x - y) <= atol + np.abs(y) * rtol


def float_cmp_all(x, y, rtol=None, atol=None):
    """Compare x and y for almost equality.

    Returns `True` if all components of `x` are almost equal to the corresponding
    components of `y`.

    See :meth:`float_cmp`.
    """
    return np.all(float_cmp(x, y, rtol, atol))


def bounded(lower, upper, x, rtol=None, atol=None):
    """Check if x is strictly in bounds (lower, upper)
    or float_compares equal to lower or upper

    Parameters
    ----------
    lower
        Lower bound
    upper
        Upper bound
    x
        value to check
    rtol
        relative tolerance for float_cmp
    atol
        absolute tolerance for float_cmp
    """
    return (lower < x < upper) or float_cmp(x, lower, rtol, atol) or float_cmp(x, upper, rtol, atol)


def contains_zero_vector(vector_array, rtol=None, atol=None):
    """returns `True` iff any vector in the array float_compares to 0s of the same dim

    Parameters
    ----------
    vector_array
        a |VectorArray| implementation
    rtol
        relative tolerance for float_cmp
    atol
        absolute tolerance for float_cmp
    """
    zero = np.zeros(vector_array.dim)

    for i in range(len(vector_array)):
        vec = vector_array[i].to_numpy()
        if float_cmp_all(vec, zero, rtol, atol):
            return True
    return False

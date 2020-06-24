# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import operator
import warnings
from functools import partial

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


@defaults('rtol', 'atol')
def compare_with_tolerance(x, y, comparison_op, rtol=1e-14, atol=1e-14):
    """ 'One-sided' Comparison x and y component-wise with given comparison op.

    For scalars we define almost equality as ::

       compare_with_tolerance(x,y) <=> op(x - y, atol + y*rtol)

    Parameters
    ----------
    x, y
        |NumPy arrays| to be compared. Have to be broadcastable to the same shape.
    comparison_op
        binary operator object, see |operator| module.
    rtol
        The relative tolerance.
    atol
        The absolute tolerance.
    """
    if comparison_op is operator.eq:
        warnings.warn('Use float_cmp for float equality tests')
    return comparison_op(x-y, atol + y * rtol)


def almost_less(x, y, rtol=None, atol=None):
    return compare_with_tolerance(x, y, operator.le, rtol, atol)
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor import defaults


def float_cmp(x, y, rtol=None, atol=None):
    '''Compare x and y component-wise for almost equality.

    For scalars we define almost equality as ::

       float_cmp(x,y) <=> |x - y| <= atol + |y|*rtol

    NB. Numpy's allclose uses the same definition but treats arrays
    containing infinities as close if the infinities are at the same
    places and all other entries are close.
    In our definition, arrays containing infinities can never be close
    which seems more appropriate in most cases.

    Parameters
    ----------
    x, y
        Arrays to be compared. Have to be broadcastable to the same shape.
    rtol
        The relative tolerance. If None, it is set to `defaults.float_cmp_tol`.
    atol
        The absolute tolerance. If None, it is set to rtol.
    '''

    rtol = rtol or defaults.float_cmp_tol
    atol = atol or rtol
    return np.abs(x - y) <= atol + np.abs(y) * rtol


def float_cmp_all(x, y, rtol=None, atol=None):
    '''Compare x and y for almost equality.

    Returns `True` if all components of `x` are almost equal to the corresponding
    components of `y`.

    See `float_cmp`.
    '''
    return np.all(float_cmp(x, y, rtol, atol))

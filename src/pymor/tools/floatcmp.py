# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

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

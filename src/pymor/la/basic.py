from __future__ import absolute_import, division, print_function

import numbers

import numpy as np
from pymor.core import defaults


def float_cmp(x, y, rtol=None, atol=None):
    '''Compare x and y for almost equality.
    If rtol == None, we set rtol = float_cmp_tol.
    If atol == None, we set atol = rtol.

    We define almost equality as

       float_cmp(x,y) <=> |x - y| <= atol*eps + |y|*rtol*eps
    '''

    rtol = rtol or defaults.float_cmp_tol
    atol = atol or rtol
    if isinstance(x, (numbers.Rational, numbers.Integral)):
        x = float(x)
    if isinstance(y, (numbers.Rational, numbers.Integral)):
        y = float(y)
    dtx = x.dtype if isinstance(x, np.ndarray) else np.dtype(type(x))
    dty = y.dtype if isinstance(y, np.ndarray) else np.dtype(type(y))
    eps = min(np.finfo(dtx).eps, np.finfo(dty).eps)
    return np.abs(x - y) <= atol * eps + np.abs(y) * rtol * eps

def float_cmp_all(x, y, rtol=None, atol=None):
    return np.all(float_cmp(x, y, rtol, atol))

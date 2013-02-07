from __future__ import absolute_import, division, print_function

import numpy as np
from pymor.core import defaults


def float_cmp(x, y, rtol=None, atol=None):
    '''Compare x and y for almost equality.
    If rtol == None, we set rtol = float_cmp_tol.
    If atol == None, we set atol = rtol.

    We define almost equality as

       float_cmp(x,y) <=> |x - y| <= atol + |y|*rtol
    '''

    rtol = rtol or defaults.float_cmp_tol
    atol = atol or rtol
    return np.abs(x - y) <= atol + np.abs(y) * rtol

def float_cmp_all(x, y, rtol=None, atol=None):
    return np.all(float_cmp(x, y, rtol, atol))

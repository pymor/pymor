# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.defaults import defaults

import numpy as np


@defaults('compact_print', sid_ignore=('compact_print',))
def format_array(array, compact_print=False):
    '''Creates a formatted string representation of a |NumPy array|.

    Parameters
    ----------
    compact_print
        If `True`, return a shorter version of string representation.

    Returns
    -------
    The string representation.
    '''
    def format_element(e):
        if e > 1e15:
            return '%(n).2e' % {'n': e}
        elif e == np.floor(e):
            return '%(n).0f' % {'n': e}
        elif e - np.floor(e) > 0.01 or e < 1000:
            return '%(n).2f' % {'n': e}
        else:
            return '%(n).2e' % {'n': e}

    if array.ndim == 0:
        return str(array.item())
    elif len(array) == 0:
        return ''
    elif len(array) == 1:
        if compact_print:
            return '[' + format_element(array[0]) + ']'
        else:
            return '[{}]'.format(array[0])
    s = '['
    for ii in np.arange(len(array) - 1):
        if compact_print:
            s += format_element(array[ii]) + ', '
        else:
            s += '{}, '.format(array[ii])
    if compact_print:
        s += format_element(array[-1]) + ']'
    else:
        s += '{}]'.format(array[-1])
    return s

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.defaults import defaults

import numpy as np


@defaults('compact_print')
def format_array(array, compact_print=False):
    """Creates a formatted string representation of a |NumPy array|.

    Parameters
    ----------
    array
        the |NumPy array| to be formatted
    compact_print
        If `True`, return a shorter version of string representation.

    Returns
    -------
    The string representation.
    """
    def format_element(e):
        if e > 1e15:
            return f'{e:.2e}'
        elif e == np.floor(e):
            return f'{e:.0f}'
        elif e - np.floor(e) > 0.01 or e < 1000:
            return f'{e:.2f}'
        else:
            return f'{e:.2e}'

    if array.ndim == 0:
        return str(array.item())
    elif len(array) == 0:
        return ''
    elif len(array) == 1:
        if compact_print:
            return '[' + format_element(array[0]) + ']'
        else:
            return f'[{array[0]}]'
    s = '['
    for ii in np.arange(len(array) - 1):
        if compact_print:
            s += format_element(array[ii]) + ', '
        else:
            s += f'{array[ii]}, '
    if compact_print:
        s += format_element(array[-1]) + ']'
    else:
        s += f'{array[-1]}]'
    return s

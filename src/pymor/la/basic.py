# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor import defaults


def induced_norm(product):
    '''The induced norm of a scalar product.

    The norm of a vector (an array of vectors) U is calculated by
    calling ::

        product.apply2(U, U, mu=mu, pairwise=True)

    In addition, negative norm squares of absolute value smaller
    than the `induced_norm_tol` |default| value are clipped to `0`.
    If the `induced_norm_raise_negative` |default| value is `True`,
    a :exc:`ValueError` exception is raised if there are still
    negative norm squares afterwards.

    Parameters
    ----------
    product
        The scalar product for which the norm is to be calculated,
        given as a linear |Operator|.

    Returns
    -------
    norm
        A function `norm(U, mu=None)` taking a |VectorArray| `U`
        as input together with the |Parameter| `mu` which is
        passed to the product.
    '''

    def norm(U, mu=None):
        norm_squared = product.apply2(U, U, mu=mu, pairwise=True)
        if defaults.induced_norm_tol > 0:
            norm_squared = np.where(np.logical_and(0 > norm_squared, norm_squared > - defaults.induced_norm_tol),
                                    0, norm_squared)
        if defaults.induced_norm_raise_negative and np.any(norm_squared < 0):
            raise ValueError('norm is negative (square = {})'.format(norm_squared))
        return np.sqrt(norm_squared)

    return norm

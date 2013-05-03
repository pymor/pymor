# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import math as m

from pymor.core import defaults


def induced_norm(product):
    '''The induced norm of a scalar product.

    The norm of a vector (an array of vectors) U is calcuated by
    calling ::

        product.apply2(U, U, mu=mu, pairwise=True)

    Parameters
    ----------
    product
        The scalar product for which the norm is to be calculated.
        Either a `DiscreteLinearOperator` or a square matrix.

    Returns
    -------
    norm
        A function `norm(U, mu=None)` taking a vector or an array of
        vectors as input together with the parameter `mu` which is
        passed to the product.
    '''

    def norm(U, mu=None):
        assert len(U) == 1
        norm_squared = product.apply2(U, U, mu=mu, pairwise=True)
        if norm_squared < 0:
            if (-norm_squared < defaults.induced_norm_tol):
                return 0
            if defaults.induced_norm_raise_negative:
                raise ValueError('norm is not negative (square = {})'.format(norm_squared))
            return 0
        return m.sqrt(norm_squared)

    return norm

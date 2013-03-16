from __future__ import absolute_import, division, print_function

import numpy as np
import math as m

from pymor.core import defaults
from pymor.discreteoperators import DiscreteOperatorInterface, GenericLinearOperator


def l2_norm(U):
    return np.sqrt(np.sum(U ** 2, axis=-1))


def induced_norm(product):
    if not isinstance(product, DiscreteOperatorInterface):
        product = GenericLinearOperator(product)

    def norm(U, mu={}):
        norm_squared = product.apply2(U, U, mu, pairwise=True)
        if norm_squared < 0:
            if (-norm_squared < defaults.induced_norm_tol):
                return 0
            if defaults.induced_norm_raise_negative:
                raise ValueError('norm is not negative (square = {})'.format(norm_squared))
            return 0
        return m.sqrt(norm_squared)

    return norm

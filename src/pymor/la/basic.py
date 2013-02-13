from __future__ import absolute_import, division, print_function

import numpy as np
import math as m

from pymor.core import defaults
from pymor.core.exceptions import AccuracyError
from pymor.tools import float_cmp_all
from pymor.discreteoperators import DiscreteOperatorInterface, GenericLinearOperator


def l2_norm(U):
    return np.sqrt(np.sum(U**2, axis=-1))


def induced_norm(product):
    if not isinstance(product, DiscreteOperatorInterface):
        product = GenericLinearOperator(product)

    def norm(U, mu={}):
        return m.sqrt(product.apply2(U, U, mu, pairwise=True))

    return norm

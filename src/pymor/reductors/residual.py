# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import ImmutableInterface
from pymor.core.logger import getLogger
from pymor.la import NumpyVectorArray, NumpyVectorSpace
from pymor.operators.basic import OperatorBase


class DummyReconstructor(ImmutableInterface):

    def reconstruct(self, U):
        return U


class GenericResidualOperator(OperatorBase):

    def __init__(self, operator, functional, product):
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.operator = operator
        self.functional = functional
        self.product = product

    def apply(self, U, ind=None, mu=None):
        V = self.operator.apply(U, ind=ind, mu=mu)
        if self.functional:
            F = self.functional.as_vector(mu)
            if len(V) > 1:
                R = F.copy(ind=[0]*len(V)) - V
            else:
                R = F - V
        else:
            R = - V
        if self.product:
            R = self.product.apply_inverse(R)
        return R


def reduce_residual(operator, functional=None, RB=None, product=None, disable_caching=True, extends=None):
    """Generic reduced basis residual reductor.
    """
    assert functional == None \
        or functional.range == NumpyVectorSpace(1) and functional.source == operator.source and functional.linear
    assert RB is None or RB in operator.source
    assert product is None or product.source == product.range == operator.range
    assert extends is None or len(extends) == 3
    if extends is not None:
        raise NotImplementedError

    logger = getLogger('pymor.reductors.reduce_residual')

    if RB is None:
        RB = operator.source.empty()

    logger.warn('Using inefficient GenericResidualOperator for residual computation')
    operator = operator.projected(RB, None)
    return GenericResidualOperator(operator, functional, product), DummyReconstructor, {}

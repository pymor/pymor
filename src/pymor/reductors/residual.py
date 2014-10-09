# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import ImmutableInterface
from pymor.core.logger import getLogger
from pymor.la import NumpyVectorArray, NumpyVectorSpace, induced_norm
from pymor.la.gram_schmidt import gram_schmidt
from pymor.operators.basic import OperatorBase, LincombOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import GenericRBReconstructor


class ResidualOperator(OperatorBase):

    def __init__(self, operator, functional):
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.operator = operator
        self.functional = functional

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
        return R


class NonProjectedResiudalOperator(ResidualOperator):

    def __init__(self, operator, functional, product):
        super(NonProjectedResiudalOperator, self).__init__(operator, functional)
        self.product = product

    def apply(self, U, ind=None, mu=None):
        R = super(NonProjectedResiudalOperator, self).apply(U, ind=ind, mu=mu)
        if self.product:
            R_riesz = self.product.apply_inverse(R)
            return R_riesz * (np.sqrt(R_riesz.dot(R, pairwise=True)) / R_riesz.l2_norm())[0]
        else:
            return R


class NonProjectedReconstructor(ImmutableInterface):

    def __init__(self, product):
        self.norm = induced_norm(product) if product else None

    def reconstruct(self, U):
        if self.norm:
            return U * (U.l2_norm() / self.norm(U))[0]
        else:
            return U


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

    residual_range = operator.range.empty()

    class CollectionError(Exception):
        def __init__(self, op):
            super(CollectionError, self).__init__()
            self.op = op

    def collect_ranges(op, functional, residual_range):
        if isinstance(op, LincombOperator):
            for o in op.operators:
                collect_ranges(o, functional, residual_range)
        elif isinstance(op, EmpiricalInterpolatedOperator):
            if hasattr(op, 'collateral_basis'):
                residual_range.append(op.collateral_basis)
        elif op.linear and not op.parametric:
            if not functional:
                residual_range.append(op.apply(RB))
            else:
                residual_range.append(op.as_vector())
        else:
            raise CollectionError(op)

    try:
        collect_ranges(operator, False, residual_range)
        collect_ranges(functional, True, residual_range)
    except CollectionError as e:
        logger.warn('Cannot compute range of {}. Evaluation will be slow.'.format(e.op))
        operator = operator.projected(RB, None)
        return (NonProjectedResiudalOperator(operator, functional, product),
                NonProjectedReconstructor(product),
                {})

    if product:
        logger.info('Computing Riesz representatives ...')
        residual_range = product.apply_inverse(residual_range)

    logger.info('Orthonormalizing ...')
    gram_schmidt(residual_range, product=product, copy=False)

    logger.info('Projecting ...')
    operator = operator.projected(RB, residual_range, product=None)  # the product always cancels out.
    functional = functional.projected(residual_range, None, product=None)

    return (ResidualOperator(operator, functional),
            GenericRBReconstructor(residual_range),
            {})

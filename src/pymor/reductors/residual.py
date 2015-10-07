# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.interfaces import ImmutableInterface
from pymor.core.logger import getLogger
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import (LincombOperator, SelectionOperator, AdjointOperator, Concatenation,
                                           induced_norm)
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import GenericRBReconstructor
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray


def reduce_residual(operator, functional=None, RB=None, product=None, extends=None):
    """Generic reduced basis residual reductor.

    Given an operator and a functional, the concatenation of residual operator
    with the Riesz isomorphism is given by::

        riesz_residual.apply(U, mu)
            == product.apply_inverse(operator.apply(U, mu) - functional.as_vector(mu))

    This reductor determines a low-dimensional subspace of image of a reduced
    basis space under `riesz_residual`, computes an orthonormal basis `residual_range`
    of this range spaces and then returns the Petrov-Galerkin projection ::

        projected_riesz_residual
            === riesz_residual.projected(range_basis=residual_range, source_basis=RB)

    of the `riesz_residual` operator. Given an reduced basis coefficient vector `u`,
    the dual norm of the residual can then be computed as ::

        projected_riesz_residual.apply(u, mu).l2_norm()

    Moreover, a `reconstructor` is provided such that ::

        reconstructor.reconstruct(projected_riesz_residual.apply(u, mu))
            == riesz_residual.apply(RB.lincomb(u), mu)

    Parameters
    ----------
    operator
        See definition of `riesz_residual`.
    functional
        See definition of `riesz_residual`. If `None`, zero right-hand side is assumed.
    RB
        |VectorArray| containing a basis of the reduced space onto which to project.
    product
        Scalar product |Operator| w.r.t. which to compute the Riesz representatives.
    extends
        Set by :meth:`~pymor.algorithms.greedy.greedy` to the result of the
        last reduction in case the basis extension was `hierarchic`. Used to prevent
        re-computation of `residual_range` basis vectors already obtained from previous
        reductions.

    Returns
    -------
    projected_riesz_residual
        See above.
    reconstructor
        See above.
    reduction_data
        Additional data produced by the reduction process. (Compare the `extends` parameter.)
    """
    assert functional is None \
        or functional.range == NumpyVectorSpace(1) and functional.source == operator.source and functional.linear
    assert RB is None or RB in operator.source
    assert product is None or product.source == product.range == operator.range
    assert extends is None or len(extends) == 3

    logger = getLogger('pymor.reductors.reduce_residual')

    if RB is None:
        RB = operator.source.empty()

    if extends and isinstance(extends[0], NonProjectedResidualOperator):
        extends = None
    if extends:
        residual_range = extends[1].RB.copy()
        residual_range_dims = list(extends[2]['residual_range_dims'])
        ind_range = range(extends[0].source.dim, len(RB))
    else:
        residual_range = operator.range.empty()
        residual_range_dims = []
        ind_range = range(-1, len(RB))

    class CollectionError(Exception):
        def __init__(self, op):
            super(CollectionError, self).__init__()
            self.op = op

    def collect_operator_ranges(op, source, ind, residual_range):
        if isinstance(op, (LincombOperator, SelectionOperator)):
            for o in op.operators:
                collect_operator_ranges(o, source, ind, residual_range)
        elif isinstance(op, EmpiricalInterpolatedOperator):
            if hasattr(op, 'collateral_basis') and ind == -1:
                residual_range.append(op.collateral_basis)
        elif isinstance(op, Concatenation):
            firstrange = op.first.range.empty()
            collect_operator_ranges(op.first, source, ind, firstrange)
            for j in range(len(firstrange)):
                collect_operator_ranges(op.second, firstrange, j, residual_range)
        elif op.linear and not op.parametric:
            if ind >= 0:
                residual_range.append(op.apply(source, ind=ind))
        else:
            raise CollectionError(op)

    def collect_functional_ranges(op, residual_range):
        if isinstance(op, (LincombOperator, SelectionOperator)):
            for o in op.operators:
                collect_functional_ranges(o, residual_range)
        elif isinstance(op, AdjointOperator):
            operator = Concatenation(op.range_product, op.operator) if op.range_product else op.operator
            collect_operator_ranges(operator, NumpyVectorArray(np.ones(1)), 0, residual_range)
        elif op.linear and not op.parametric:
            residual_range.append(op.as_vector())
        else:
            raise CollectionError(op)

    for i in ind_range:
        logger.info('Computing residual range for basis vector {}...'.format(i))
        new_residual_range = operator.range.empty()
        try:
            if i == -1:
                collect_functional_ranges(functional, new_residual_range)
            collect_operator_ranges(operator, RB, i, new_residual_range)
        except CollectionError as e:
            logger.warn('Cannot compute range of {}. Evaluation will be slow.'.format(e.op))
            operator = operator.projected(None, RB)
            return (NonProjectedResidualOperator(operator, functional, product),
                    NonProjectedReconstructor(product),
                    {})

        if product:
            logger.info('Computing Riesz representatives for basis vector {}...'.format(i))
            new_residual_range = product.apply_inverse(new_residual_range)

        gram_schmidt_offset = len(residual_range)
        residual_range.append(new_residual_range)
        logger.info('Orthonormalizing ...')
        gram_schmidt(residual_range, offset=gram_schmidt_offset, product=product, copy=False)
        residual_range_dims.append(len(residual_range))

    logger.info('Projecting ...')
    operator = operator.projected(residual_range, RB, product=None)  # the product always cancels out.
    functional = functional.projected(None, residual_range, product=None)

    return (ResidualOperator(operator, functional),
            GenericRBReconstructor(residual_range),
            {'residual_range_dims': residual_range_dims})


class ResidualOperator(OperatorBase):
    """Returned by :func:`reduce_residual`."""

    def __init__(self, operator, functional, name=None):
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.operator = operator
        self.functional = functional
        self.functional_vector = functional.as_vector() if functional and not functional.parametric else None
        self.name = name

    def apply(self, U, ind=None, mu=None):
        V = self.operator.apply(U, ind=ind, mu=mu)
        if self.functional:
            F = self.functional_vector or self.functional.as_vector(mu)
            if len(V) > 1:
                V.axpy(-1., F, x_ind=[0]*len(V))
            else:
                V.axpy(-1., F)
        return V

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return ResidualOperator(self.operator.projected_to_subbasis(dim_range, dim_source),
                                self.functional.projected_to_subbasis(None, dim_range),
                                name=name)


class NonProjectedResidualOperator(ResidualOperator):
    """Returned by :func:`reduce_residual`.

    Not to be used directly.
    """

    def __init__(self, operator, functional, product):
        super(NonProjectedResidualOperator, self).__init__(operator, functional)
        self.product = product

    def apply(self, U, ind=None, mu=None):
        R = super(NonProjectedResidualOperator, self).apply(U, ind=ind, mu=mu)
        if self.product:
            R_riesz = self.product.apply_inverse(R)
            return R_riesz * (np.sqrt(R_riesz.dot(R)) / R_riesz.l2_norm())[0]
        else:
            return R

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        product = self.product.projected_to_subbasis(dim_range, dim_range) if self.product is not None else None
        return ResidualOperator(self.operator.projected_to_subbasis(dim_range, dim_source),
                                self.functional.projected_to_subbasis(None, dim_range),
                                product,
                                name=name)


class NonProjectedReconstructor(ImmutableInterface):
    """Returned by :func:`reduce_residual`.

    Not to be used directly.
    """

    def __init__(self, product):
        self.norm = induced_norm(product) if product else None

    def reconstruct(self, U):
        if self.norm:
            return U * (U.l2_norm() / self.norm(U))[0]
        else:
            return U

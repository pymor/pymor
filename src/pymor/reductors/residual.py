# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.image import estimate_image_hierarchical
from pymor.core.interfaces import ImmutableInterface
from pymor.core.exceptions import ImageCollectionError
from pymor.core.logger import getLogger
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import induced_norm
from pymor.reductors.basic import GenericRBReconstructor
from pymor.vectorarrays.numpy import NumpyVectorSpace


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

    logger = getLogger('pymor.reductors.residual.reduce_residual')

    if RB is None:
        RB = operator.source.empty()

    if extends and isinstance(extends[0], NonProjectedResidualOperator):
        extends = None
    if extends:
        residual_range = extends[1].RB
        residual_range_dims = list(extends[2]['residual_range_dims'])
    else:
        residual_range = operator.range.empty()
        residual_range_dims = []

    logger.info('Estimating residual range ...')
    try:
        residual_range, residual_range_dims = \
            estimate_image_hierarchical([operator], [functional], RB, (residual_range, residual_range_dims),
                                        orthonormalize=True, product=product, riesz_representatives=True)
    except ImageCollectionError as e:
        logger.warn('Cannot compute range of {}. Evaluation will be slow.'.format(e.op))
        operator = operator.projected(None, RB)
        return (NonProjectedResidualOperator(operator, functional, product),
                NonProjectedReconstructor(product),
                {})

    logger.info('Projecting residual operator ...')
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

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return self.with_(operator=self.operator.projected_to_subbasis(None, dim_source))


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

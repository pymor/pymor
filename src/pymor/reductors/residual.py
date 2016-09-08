# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.image import estimate_image_hierarchical
from pymor.core.interfaces import ImmutableInterface
from pymor.core.exceptions import ImageCollectionError
from pymor.core.logger import getLogger
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import induced_norm
from pymor.reductors.basic import GenericRBReconstructor
from pymor.vectorarrays.numpy import NumpyVectorSpace


def reduce_residual(operator, rhs=None, RB=None, rhs_is_functional=True, product=None, extends=None):
    """Generic reduced basis residual reductor.

    Given an operator and a right-hand side, the residual is given by::

        residual.apply(U, mu) == operator.apply(U, mu) - rhs.as_vector(mu)

    When the rhs is a functional we are usually interested in the Riesz representative
    of the residual::

        residual.apply(U, mu)
            == product.apply_inverse(operator.apply(U, mu) - rhs.as_vector(mu))

    Given a basis `RB` of a subspace of the source space of `operator`, this method
    uses :func:`~pymor.algorithms.image.estimate_image_hierarchical` to determine
    a low-dimensional subspace containing the image of the subspace under
    `residual` (resp. `riesz_residual`), computes an orthonormal basis
    `residual_range` for this range space and then returns the Petrov-Galerkin projection ::

        projected_residual
            === residual.projected(range_basis=residual_range, source_basis=RB)

    of the residual operator. Given an reduced basis coefficient vector `u`, w.r.t.
    `RB`, the (dual) norm of the residual can then be computed as ::

        projected_residual.apply(u, mu).l2_norm()

    Moreover, a `reconstructor` is provided such that ::

        reconstructor.reconstruct(projected_residual.apply(u, mu))
            == residual.apply(RB.lincomb(u), mu)

    Parameters
    ----------
    operator
        See definition of `residual`.
    rhs
        See definition of `residual`. If `None`, zero right-hand side is assumed.
    rhs_is_functional
        Set this to `True` when `rhs` is a |Functional|.
    RB
        |VectorArray| containing a basis of the reduced space onto which to project.
    product
        Inner product |Operator| w.r.t. which to compute the Riesz representatives
        in case `rhs_is_functional` is `True`. When `product` is `None`, no Riesz
        representatives are computed
    extends
        Set by :meth:`~pymor.algorithms.greedy.greedy` to the result of the
        last reduction in case the basis extension was `hierarchic` (used to prevent
        re-computation of `residual_range` basis vectors already obtained from previous
        reductions).

    Returns
    -------
    projected_residual
        See above.
    reconstructor
        See above.
    reduction_data
        Additional data produced by the reduction process (compare the `extends` parameter).
    """
    assert rhs is None \
        or rhs_is_functional and (rhs.range == NumpyVectorSpace(1) and rhs.source == operator.range and rhs.linear) \
        or not rhs_is_functional and (rhs.source == NumpyVectorSpace(1) and rhs.range == operator.range and rhs.linear)
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

    with logger.block('Estimating residual range ...'):
        try:
            residual_range, residual_range_dims = \
                estimate_image_hierarchical([operator], [rhs], RB, (residual_range, residual_range_dims),
                                            orthonormalize=True, product=product,
                                            riesz_representatives=rhs_is_functional)
        except ImageCollectionError as e:
            logger.warn('Cannot compute range of {}. Evaluation will be slow.'.format(e.op))
            operator = operator.projected(None, RB)
            return (NonProjectedResidualOperator(operator, rhs, rhs_is_functional, product),
                    NonProjectedReconstructor(product),
                    {})

    with logger.block('Projecting residual operator ...'):
        if rhs_is_functional:
            operator = operator.projected(residual_range, RB, product=None)  # the product cancels out.
            rhs = rhs.projected(None, residual_range, product=None)
        else:
            operator = operator.projected(residual_range, RB, product=product)
            rhs = rhs.projected(residual_range, None, product=product)

    return (ResidualOperator(operator, rhs, rhs_is_functional),
            GenericRBReconstructor(residual_range),
            {'residual_range_dims': residual_range_dims})


class ResidualOperator(OperatorBase):
    """Returned by :func:`reduce_residual`."""

    def __init__(self, operator, rhs, rhs_is_functional=True, name=None):
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.operator = operator
        self.rhs = rhs
        self.rhs_vector = rhs.as_vector() if rhs and not rhs.parametric else None
        self.rhs_is_functional = rhs_is_functional
        self.name = name

    def apply(self, U, ind=None, mu=None):
        V = self.operator.apply(U, ind=ind, mu=mu)
        if self.rhs:
            F = self.rhs_vector or self.rhs.as_vector(mu)
            if len(V) > 1:
                V.axpy(-1., F, x_ind=[0]*len(V))
            else:
                V.axpy(-1., F)
        return V

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        if self.rhs_is_functional:
            rhs = self.rhs.projected_to_subbasis(None, dim_range)
        else:
            rhs = self.rhs.projected_to_subbasis(dim_range, None)
        return ResidualOperator(self.operator.projected_to_subbasis(dim_range, dim_source), rhs,
                                self.rhs_is_functional, name=name)


class NonProjectedResidualOperator(ResidualOperator):
    """Returned by :func:`reduce_residual`.

    Not to be used directly.
    """

    def __init__(self, operator, rhs, rhs_is_functional, product):
        super().__init__(operator, rhs, rhs_is_functional)
        self.product = product

    def apply(self, U, ind=None, mu=None):
        R = super().apply(U, ind=ind, mu=mu)
        if self.product:
            if self.rhs_is_functional:
                R_riesz = self.product.apply_inverse(R)
                return R_riesz * (np.sqrt(R_riesz.dot(R)) / R_riesz.l2_norm())[0]
            else:
                return R * (np.sqrt(self.product.pairwise_apply2(R, R)) / R.l2_norm())[0]
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


def reduce_implicit_euler_residual(operator, mass, dt, functional=None, RB=None, product=None, extends=None):
    """Reduced basis residual reductor with mass operator for implicit Euler timestepping.

    Given an operator, mass and a functional, the concatenation of residual operator
    with the Riesz isomorphism is given by::

        riesz_residual.apply(U, U_old, mu)
            == product.apply_inverse(operator.apply(U, mu) + 1/dt*mass.apply(U, mu) - 1/dt*mass.apply(U_old, mu)
               - functional.as_vector(mu))

    This reductor determines a low-dimensional subspace of the image of a reduced
    basis space under `riesz_residual` using :func:`~pymor.algorithms.image.estimate_image_hierarchical`,
    computes an orthonormal basis `residual_range` of this range space and then
    returns the Petrov-Galerkin projection ::

        projected_riesz_residual
            === riesz_residual.projected(range_basis=residual_range, source_basis=RB)

    of the `riesz_residual` operator. Given reduced basis coefficient vectors `u` and `u_old`,
    the dual norm of the residual can then be computed as ::

        projected_riesz_residual.apply(u, u_old, mu).l2_norm()

    Moreover, a `reconstructor` is provided such that ::

        reconstructor.reconstruct(projected_riesz_residual.apply(u, u_old, mu))
            == riesz_residual.apply(RB.lincomb(u), RB.lincomb(u_old), mu)

    Parameters
    ----------
    operator
        See definition of `riesz_residual`.
    mass
        The mass operator. See definition of `riesz_residual`.
    dt
        The time step size. See definition of `riesz_residual`.
    functional
        See definition of `riesz_residual`. If `None`, zero right-hand side is assumed.
    RB
        |VectorArray| containing a basis of the reduced space onto which to project.
    product
        Inner product |Operator| w.r.t. which to compute the Riesz representatives.
    extends
        Set by :meth:`~pymor.algorithms.greedy.greedy` to the result of the
        last reduction in case the basis extension was `hierarchic` (used to prevent
        re-computation of `residual_range` basis vectors already obtained from previous
        reductions).

    Returns
    -------
    projected_riesz_residual
        See above.
    reconstructor
        See above.
    reduction_data
        Additional data produced by the reduction process (compare the `extends` parameter).
    """
    assert functional is None \
        or functional.range == NumpyVectorSpace(1) and functional.source == operator.source and functional.linear
    assert RB is None or RB in operator.source
    assert product is None or product.source == product.range == operator.range
    assert extends is None or len(extends) == 3

    logger = getLogger('pymor.reductors.residual.reduce_implicit_euler_residual')

    if RB is None:
        RB = operator.source.empty()

    if extends and isinstance(extends[0], NonProjectedImplicitEulerResidualOperator):
        extends = None
    if extends:
        residual_range = extends[1].RB
        residual_range_dims = list(extends[2]['residual_range_dims'])
    else:
        residual_range = operator.range.empty()
        residual_range_dims = []

    with logger.block('Estimating residual range ...'):
        try:
            residual_range, residual_range_dims = \
                estimate_image_hierarchical([operator, mass], [functional], RB, (residual_range, residual_range_dims),
                                            orthonormalize=True, product=product, riesz_representatives=True)
        except ImageCollectionError as e:
            logger.warn('Cannot compute range of {}. Evaluation will be slow.'.format(e.op))
            operator = operator.projected(None, RB)
            mass = mass.projected(None, RB)
            return (NonProjectedImplicitEulerResidualOperator(operator, mass, functional, dt, product),
                    NonProjectedReconstructor(product),
                    {})

    with logger.block('Projecting residual operator ...'):
        operator = operator.projected(residual_range, RB, product=None)  # the product always cancels out.
        mass = mass.projected(residual_range, RB, product=None)
        functional = functional.projected(None, residual_range, product=None)

    return (ImplicitEulerResidualOperator(operator, mass, functional, dt),
            GenericRBReconstructor(residual_range),
            {'residual_range_dims': residual_range_dims})


class ImplicitEulerResidualOperator(OperatorBase):
    """Returned by :func:`reduce_implicit_euler_residual`."""

    def __init__(self, operator, mass, functional, dt, name=None):
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.operator = operator
        self.mass = mass
        self.functional = functional
        self.functional_vector = functional.as_vector() if functional and not functional.parametric else None
        self.dt = dt
        self.name = name

    def apply(self, U, U_old, ind, ind_old=None, mu=None):
        V = self.operator.apply(U, ind=ind, mu=mu)
        V.axpy(1./self.dt, self.mass.apply(U, ind=ind, mu=mu))
        V.axpy(-1./self.dt, self.mass.apply(U_old, ind=ind_old, mu=mu))
        if self.functional:
            F = self.functional_vector or self.functional.as_vector(mu)
            if len(V) > 1:
                V.axpy(-1., F, x_ind=[0]*len(V))
            else:
                V.axpy(-1., F)
        return V

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return ImplicitEulerResidualOperator(self.operator.projected_to_subbasis(dim_range, dim_source),
                                             self.mass.projected_to_subbasis(dim_range, dim_source),
                                             self.functional.projected_to_subbasis(None, dim_range),
                                             self.dt,
                                             name=name)


class NonProjectedImplicitEulerResidualOperator(ImplicitEulerResidualOperator):
    """Returned by :func:`reduce_implicit_euler_residual`.

    Not to be used directly.
    """

    def __init__(self, operator, mass, functional, dt, product):
        super().__init__(operator, mass, functional, dt)
        self.product = product

    def apply(self, U, U_old, ind=None, ind_old=None, mu=None):
        R = super().apply(U, U_old, ind=ind, ind_old=ind_old, mu=mu)
        if self.product:
            R_riesz = self.product.apply_inverse(R)
            return R_riesz * (np.sqrt(R_riesz.dot(R)) / R_riesz.l2_norm())[0]
        else:
            return R

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return self.with_(operator=self.operator.projected_to_subbasis(None, dim_source),
                          mass=self.mass.projected_to_subbasis(None, dim_source))

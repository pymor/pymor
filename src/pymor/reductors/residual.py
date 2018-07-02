# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.image import estimate_image_hierarchical
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.core.interfaces import BasicInterface
from pymor.core.exceptions import ImageCollectionError
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import induced_norm


class ResidualReductor(BasicInterface):
    """Generic reduced basis residual reductor.

    Given an operator and a right-hand side, the residual is given by::

        residual.apply(U, mu) == operator.apply(U, mu) - rhs.as_vector(mu)

    When the rhs is a functional we are interested in the Riesz representative
    of the residual::

        residual.apply(U, mu)
            == product.apply_inverse(operator.apply(U, mu) - rhs.as_vector(mu))

    Given a basis `RB` of a subspace of the source space of `operator`, this reductor
    uses :func:`~pymor.algorithms.image.estimate_image_hierarchical` to determine
    a low-dimensional subspace containing the image of the subspace under
    `residual` (resp. `riesz_residual`), computes an orthonormal basis
    `residual_range` for this range space and then returns the Petrov-Galerkin projection ::

        projected_residual
            == project(residual, range_basis=residual_range, source_basis=RB)

    of the residual operator. Given a reduced basis coefficient vector `u`, w.r.t.
    `RB`, the (dual) norm of the residual can then be computed as ::

        projected_residual.apply(u, mu).l2_norm()

    Moreover, a `reconstruct` method is provided such that ::

        residual_reductor.reconstruct(projected_residual.apply(u, mu))
            == residual.apply(RB.lincomb(u), mu)

    Parameters
    ----------
    RB
        |VectorArray| containing a basis of the reduced space onto which to project.
    operator
        See definition of `residual`.
    rhs
        See definition of `residual`. If `None`, zero right-hand side is assumed.
    product
        Inner product |Operator| w.r.t. which to orthonormalize and w.r.t. which to
        compute the Riesz representatives in case `rhs` is a functional.
    """

    def __init__(self, RB, operator, rhs=None, product=None):
        assert RB in operator.source
        assert rhs is None \
            or (rhs.range.is_scalar and rhs.source == operator.range and rhs.linear) \
            or (rhs.source.is_scalar and rhs.range == operator.range and rhs.linear)
        assert product is None or product.source == product.range == operator.range

        self.RB = RB
        self.operator = operator
        self.rhs = rhs
        self.product = product
        self.residual_range = operator.range.empty()
        self.residual_range_dims = []

    def reduce(self):
        # Note that it is possible that rhs.source == rhs.range, nameley if both
        # are one-dimensional NumpyVectorSpaces which agree with the range of
        # operator. Usually, this should not happen, since at least one of these
        # spaces should have an id which is different from the id of operator.range.
        # However, even if it happens but rhs is actually a vector, we are on
        # the safe side, since first computing the Riesz representatives does not
        # change anything in one-dimensional spaces, and it does not matter whether
        # we project from the left or from the right.
        rhs_is_functional = (self.rhs.source == self.operator.range)

        if self.residual_range is not False:
            with self.logger.block('Estimating residual range ...'):
                try:
                    self.residual_range, self.residual_range_dims = \
                        estimate_image_hierarchical([self.operator], [self.rhs.T if rhs_is_functional else self.rhs],
                                                    self.RB,
                                                    (self.residual_range, self.residual_range_dims),
                                                    orthonormalize=True, product=self.product,
                                                    riesz_representatives=rhs_is_functional)
                except ImageCollectionError as e:
                    self.logger.warning('Cannot compute range of {}. Evaluation will be slow.'.format(e.op))
                    self.residual_range = False

        if self.residual_range is False:
            operator = project(self.operator, None, self.RB)
            return NonProjectedResidualOperator(operator, self.rhs, rhs_is_functional, self.product)

        with self.logger.block('Projecting residual operator ...'):
            if rhs_is_functional:
                operator = project(self.operator, self.residual_range, self.RB, product=None)  # the product cancels out.
                rhs = project(self.rhs, None, self.residual_range, product=None)
            else:
                operator = project(self.operator, self.residual_range, self.RB, product=self.product)
                rhs = project(self.rhs, self.residual_range, None, product=self.product)

        return ResidualOperator(operator, rhs, rhs_is_functional)

    def reconstruct(self, u):
        """Reconstruct high-dimensional residual vector from reduced vector `u`."""
        if self.residual_range is False:
            if self.product:
                norm = induced_norm(self.product)
                return u * (u.l2_norm() / norm(u))[0]
            else:
                return u
        else:
            return self.residual_range[:u.dim].lincomb(u.to_numpy())


class ResidualOperator(OperatorBase):
    """Instantiated by :class:`ResidualReductor`."""

    def __init__(self, operator, rhs, rhs_is_functional=True, name=None):
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.operator = operator
        self.rhs = rhs
        self.rhs_vector = rhs.as_vector(space=operator.range) if rhs and not rhs.parametric else None
        self.rhs_is_functional = rhs_is_functional
        self.name = name

    def apply(self, U, mu=None):
        V = self.operator.apply(U, mu=mu)
        if self.rhs:
            F = self.rhs_vector or self.rhs.as_vector(mu, space=self.operator.range)
            if len(V) > 1:
                V -= F[[0]*len(V)]
            else:
                V -= F
        return V

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        if self.rhs_is_functional:
            rhs = project_to_subbasis(self.rhs, None, dim_range)
        else:
            rhs = project_to_subbasis(self.rhs, dim_range, None)
        return ResidualOperator(project_to_subbasis(self.operator, dim_range, dim_source), rhs,
                                self.rhs_is_functional, name=name)


class NonProjectedResidualOperator(ResidualOperator):
    """Instantiated by :class:`ResidualReductor`.

    Not to be used directly.
    """

    def __init__(self, operator, rhs, rhs_is_functional, product):
        super().__init__(operator, rhs, rhs_is_functional)
        self.product = product

    def apply(self, U, mu=None):
        R = super().apply(U, mu=mu)
        if self.product:
            if self.rhs_is_functional:
                R_riesz = self.product.apply_inverse(R)
                return R_riesz * (np.sqrt(R_riesz.dot(R)) / R_riesz.l2_norm())[0]
            else:
                return R * (np.sqrt(self.product.pairwise_apply2(R, R)) / R.l2_norm())[0]
        else:
            return R

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return self.with_(operator=project_to_subbasis(self.operator, None, dim_source))


class ImplicitEulerResidualReductor(BasicInterface):
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
            == riesz_residual.projected(range_basis=residual_range, source_basis=RB)

    of the `riesz_residual` operator. Given reduced basis coefficient vectors `u` and `u_old`,
    the dual norm of the residual can then be computed as ::

        projected_riesz_residual.apply(u, u_old, mu).l2_norm()

    Moreover, a `reconstruct` method is provided such that ::

        residual_reductor.reconstruct(projected_riesz_residual.apply(u, u_old, mu))
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
    """
    def __init__(self, RB, operator, mass, dt, functional=None, product=None):
        assert RB in operator.source
        assert functional is None \
            or functional.range.is_scalar and functional.source == operator.source and functional.linear
        assert product is None or product.source == product.range == operator.range

        self.RB = RB
        self.operator = operator
        self.mass = mass
        self.dt = dt
        self.functional = functional
        self.product = product
        self.residual_range = operator.range.empty()
        self.residual_range_dims = []

    def reduce(self):
        if self.residual_range is not False:
            with self.logger.block('Estimating residual range ...'):
                try:
                    self.residual_range, self.residual_range_dims = \
                        estimate_image_hierarchical([self.operator, self.mass], [self.functional.T],
                                                    self.RB,
                                                    (self.residual_range, self.residual_range_dims),
                                                    orthonormalize=True, product=self.product,
                                                    riesz_representatives=True)
                except ImageCollectionError as e:
                    self.logger.warning('Cannot compute range of {}. Evaluation will be slow.'.format(e.op))
                    self.residual_range = False

        if self.residual_range is False:
            operator = project(self.operator, None, self.RB)
            mass = project(self.mass, None, self.RB)
            return NonProjectedImplicitEulerResidualOperator(operator, mass, self.functional, self.dt, self.product)

        with self.logger.block('Projecting residual operator ...'):
            operator = project(self.operator, self.residual_range, self.RB, product=None)  # the product always cancels out.
            mass = project(self.mass, self.residual_range, self.RB, product=None)
            functional = project(self.functional, None, self.residual_range, product=None)

        return ImplicitEulerResidualOperator(operator, mass, functional, self.dt)

    def reconstruct(self, u):
        """Reconstruct high-dimensional residual vector from reduced vector `u`."""
        if self.residual_range is False:
            if self.product:
                norm = induced_norm(self.product)
                return u * (u.l2_norm() / norm(u))[0]
            else:
                return u
        else:
            return self.residual_range[:u.dim].lincomb(u.to_numpy())


class ImplicitEulerResidualOperator(OperatorBase):
    """Instantiated by :class:`ImplicitEulerResidualReductor`."""

    def __init__(self, operator, mass, functional, dt, name=None):
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.operator = operator
        self.mass = mass
        self.functional = functional
        self.functional_vector = functional.as_source_array() if functional and not functional.parametric else None
        self.dt = dt
        self.name = name

    def apply(self, U, U_old, mu=None):
        V = self.operator.apply(U, mu=mu)
        V.axpy(1./self.dt, self.mass.apply(U, mu=mu))
        V.axpy(-1./self.dt, self.mass.apply(U_old, mu=mu))
        if self.functional:
            F = self.functional_vector or self.functional.as_source_array(mu)
            if len(V) > 1:
                V -= F[[0]*len(V)]
            else:
                V -= F
        return V

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return ImplicitEulerResidualOperator(project_to_subbasis(self.operator, dim_range, dim_source),
                                             project_to_subbasis(self.mass, dim_range, dim_source),
                                             project_to_subbasis(self.functional, None, dim_range),
                                             self.dt,
                                             name=name)


class NonProjectedImplicitEulerResidualOperator(ImplicitEulerResidualOperator):
    """Instantiated by :class:`ImplicitEulerResidualReductor`.

    Not to be used directly.
    """

    def __init__(self, operator, mass, functional, dt, product):
        super().__init__(operator, mass, functional, dt)
        self.product = product

    def apply(self, U, U_old, mu=None):
        R = super().apply(U, U_old, mu=mu)
        if self.product:
            R_riesz = self.product.apply_inverse(R)
            return R_riesz * (np.sqrt(R_riesz.pairwise_dot(R)) / R_riesz.l2_norm())[0]
        else:
            return R

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return self.with_(operator=project_to_subbasis(self.operator, None, dim_source),
                          mass=project_to_subbasis(self.mass, None, dim_source))

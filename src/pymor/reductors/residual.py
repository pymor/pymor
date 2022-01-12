# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.image import estimate_image_hierarchical
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.core.base import BasicObject
from pymor.core.exceptions import ImageCollectionError
from pymor.operators.constructions import ZeroOperator
from pymor.operators.interface import Operator


class ResidualReductor(BasicObject):
    """Generic reduced basis residual reductor.

    Given an operator and a right-hand side, the residual is given by::

        residual.apply(U, mu) == operator.apply(U, mu) - rhs.as_range_array(mu)

    When operator maps to functionals instead of vectors, we are interested in the Riesz
    representative of the residual::

        residual.apply(U, mu)
            == product.apply_inverse(operator.apply(U, mu) - rhs.as_range_array(mu))

    Given a basis `RB` of a subspace of the source space of `operator`, this reductor
    uses :func:`~pymor.algorithms.image.estimate_image_hierarchical` to determine
    a low-dimensional subspace containing the image of the subspace under
    `residual` (resp. `riesz_residual`), computes an orthonormal basis
    `residual_range` for this range space and then returns the Petrov-Galerkin projection ::

        projected_residual
            == project(residual, range_basis=residual_range, source_basis=RB)

    of the residual operator. Given a reduced basis coefficient vector `u`, w.r.t.
    `RB`, the (dual) norm of the residual can then be computed as ::

        projected_residual.apply(u, mu).norm()

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
        compute the Riesz representatives in case `operator` maps to functionals.
    riesz_representatives
        If `True` compute the Riesz representative of the residual.
    """

    def __init__(self, RB, operator, rhs=None, product=None, riesz_representatives=False):
        assert RB in operator.source
        assert rhs is None \
            or (rhs.source.is_scalar and rhs.range == operator.range and rhs.linear)
        assert product is None or product.source == product.range == operator.range

        self.__auto_init(locals())
        self.residual_range = operator.range.empty()
        self.residual_range_dims = []

    def reduce(self):
        if self.residual_range is not False:
            with self.logger.block('Estimating residual range ...'):
                try:
                    self.residual_range, self.residual_range_dims = \
                        estimate_image_hierarchical([self.operator], [self.rhs],
                                                    self.RB,
                                                    (self.residual_range, self.residual_range_dims),
                                                    orthonormalize=True, product=self.product,
                                                    riesz_representatives=self.riesz_representatives)
                except ImageCollectionError as e:
                    self.logger.warning(f'Cannot compute range of {e.op}. Evaluation will be slow.')
                    self.residual_range = False

        if self.residual_range is False:
            operator = project(self.operator, None, self.RB)
            return NonProjectedResidualOperator(operator, self.rhs, self.riesz_representatives, self.product)

        with self.logger.block('Projecting residual operator ...'):
            if self.riesz_representatives:
                operator = project(self.operator, self.residual_range, self.RB, product=None)  # the product cancels out
                rhs = project(self.rhs, self.residual_range, None, product=None)
            else:
                operator = project(self.operator, self.residual_range, self.RB, product=self.product)
                rhs = project(self.rhs, self.residual_range, None, product=self.product)

        return ResidualOperator(operator, rhs)

    def reconstruct(self, u):
        """Reconstruct high-dimensional residual vector from reduced vector `u`."""
        if self.residual_range is False:
            if self.product:
                return u * (u.norm() / u.norm(self.product))[0]
            else:
                return u
        else:
            return self.residual_range[:u.dim].lincomb(u.to_numpy())


class ResidualOperator(Operator):
    """Instantiated by :class:`ResidualReductor`."""

    def __init__(self, operator, rhs, name=None):
        self.__auto_init(locals())
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.rhs_vector = rhs.as_range_array() if rhs and not rhs.parametric else None

    def apply(self, U, mu=None):
        V = self.operator.apply(U, mu=mu)
        if self.rhs:
            F = self.rhs_vector or self.rhs.as_range_array(mu)
            if len(V) > 1:
                V -= F[[0]*len(V)]
            else:
                V -= F
        return V

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return ResidualOperator(project_to_subbasis(self.operator, dim_range, dim_source),
                                project_to_subbasis(self.rhs, dim_range, None),
                                name=name)


class NonProjectedResidualOperator(ResidualOperator):
    """Instantiated by :class:`ResidualReductor`.

    Not to be used directly.
    """

    def __init__(self, operator, rhs, riesz_representatives, product):
        super().__init__(operator, rhs)
        self.__auto_init(locals())

    def apply(self, U, mu=None):
        R = super().apply(U, mu=mu)
        if self.product:
            if self.riesz_representatives:
                R_riesz = self.product.apply_inverse(R)
                # divide by norm, except when norm is zero:
                inversel2 = 1./R_riesz.norm()
                inversel2 = np.nan_to_num(inversel2)
                R_riesz.scal(np.sqrt(R_riesz.pairwise_inner(R)) * inversel2)
                return R_riesz
            else:
                # divide by norm, except when norm is zero:
                inversel2 = 1./R.norm()
                inversel2 = np.nan_to_num(inversel2)
                R.scal(np.sqrt(self.product.pairwise_apply2(R, R)) * inversel2)
                return R
        else:
            return R

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return self.with_(operator=project_to_subbasis(self.operator, None, dim_source))


class ImplicitEulerResidualReductor(BasicObject):
    """Reduced basis residual reductor with mass operator for implicit Euler timestepping.

    Given an operator, mass and a functional, the concatenation of residual operator
    with the Riesz isomorphism is given by::

        riesz_residual.apply(U, U_old, mu)
            == product.apply_inverse(operator.apply(U, mu) + 1/dt*mass.apply(U, mu)
                - 1/dt*mass.apply(U_old, mu) - rhs.as_vector(mu))

    This reductor determines a low-dimensional subspace of the image of a reduced basis space under
    `riesz_residual` using :func:`~pymor.algorithms.image.estimate_image_hierarchical`, computes an
    orthonormal basis `residual_range` of this range space and then returns the Petrov-Galerkin
    projection ::

        projected_riesz_residual
            == riesz_residual.projected(range_basis=residual_range, source_basis=RB)

    of the `riesz_residual` operator. Given reduced basis coefficient vectors `u` and `u_old`,
    the dual norm of the residual can then be computed as ::

        projected_riesz_residual.apply(u, u_old, mu).norm()

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
    rhs
        See definition of `riesz_residual`. If `None`, zero right-hand side is assumed.
    RB
        |VectorArray| containing a basis of the reduced space onto which to project.
    product
        Inner product |Operator| w.r.t. which to compute the Riesz representatives.
    """

    def __init__(self, RB, operator, mass, dt, rhs=None, product=None):
        assert RB in operator.source
        assert rhs.source.is_scalar and rhs.range == operator.range and rhs.linear
        assert product is None or product.source == product.range == operator.range

        self.__auto_init(locals())
        self.residual_range = operator.range.empty()
        self.residual_range_dims = []

    def reduce(self):
        if self.residual_range is not False:
            with self.logger.block('Estimating residual range ...'):
                try:
                    self.residual_range, self.residual_range_dims = \
                        estimate_image_hierarchical([self.operator, self.mass], [self.rhs],
                                                    self.RB,
                                                    (self.residual_range, self.residual_range_dims),
                                                    orthonormalize=True, product=self.product,
                                                    riesz_representatives=True)
                except ImageCollectionError as e:
                    self.logger.warning(f'Cannot compute range of {e.op}. Evaluation will be slow.')
                    self.residual_range = False

        if self.residual_range is False:
            operator = project(self.operator, None, self.RB)
            mass = project(self.mass, None, self.RB)
            return NonProjectedImplicitEulerResidualOperator(operator, mass, self.rhs, self.dt, self.product)

        with self.logger.block('Projecting residual operator ...'):
            # the product always cancels out
            operator = project(self.operator, self.residual_range, self.RB, product=None)
            mass = project(self.mass, self.residual_range, self.RB, product=None)
            rhs = project(self.rhs, self.residual_range, None, product=None)

        return ImplicitEulerResidualOperator(operator, mass, rhs, self.dt)

    def reconstruct(self, u):
        """Reconstruct high-dimensional residual vector from reduced vector `u`."""
        if self.residual_range is False:
            if self.product:
                return u * (u.norm() / u.norm(self.product))[0]
            else:
                return u
        else:
            return self.residual_range[:u.dim].lincomb(u.to_numpy())


class ImplicitEulerResidualOperator(Operator):
    """Instantiated by :class:`ImplicitEulerResidualReductor`."""

    def __init__(self, operator, mass, rhs, dt, name=None):
        self.__auto_init(locals())
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.rhs_vector = rhs.as_range_array() if not rhs.parametric else None

    def apply(self, U, U_old, mu=None):
        V = self.operator.apply(U, mu=mu)
        V.axpy(1./self.dt, self.mass.apply(U, mu=mu))
        V.axpy(-1./self.dt, self.mass.apply(U_old, mu=mu))
        if not isinstance(self.rhs, ZeroOperator):
            F = self.rhs_vector or self.rhs.as_range_array(mu)
            if len(V) > 1:
                V -= F[[0]*len(V)]
            else:
                V -= F
        return V

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return ImplicitEulerResidualOperator(project_to_subbasis(self.operator, dim_range, dim_source),
                                             project_to_subbasis(self.mass, dim_range, dim_source),
                                             project_to_subbasis(self.rhs, dim_range, None),
                                             self.dt,
                                             name=name)


class NonProjectedImplicitEulerResidualOperator(ImplicitEulerResidualOperator):
    """Instantiated by :class:`ImplicitEulerResidualReductor`.

    Not to be used directly.
    """

    def __init__(self, operator, mass, rhs, dt, product):
        super().__init__(operator, mass, rhs, dt)
        self.product = product

    def apply(self, U, U_old, mu=None):
        R = super().apply(U, U_old, mu=mu)
        if self.product:
            R_riesz = self.product.apply_inverse(R)
            # divide by norm, except when norm is zero:
            inversel2 = 1./R_riesz.norm()
            inversel2 = np.nan_to_num(inversel2)
            R_riesz.scal(np.sqrt(R_riesz.pairwise_inner(R)) * inversel2)
            return R_riesz
        else:
            return R

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        return self.with_(operator=project_to_subbasis(self.operator, None, dim_source),
                          mass=project_to_subbasis(self.mass, None, dim_source))

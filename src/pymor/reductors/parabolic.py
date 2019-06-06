# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.reductors.basic import InstationaryRBReductor
from pymor.reductors.residual import ResidualReductor, ImplicitEulerResidualReductor
from pymor.operators.constructions import IdentityOperator
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper


class ParabolicRBReductor(InstationaryRBReductor):
    r"""Reduced Basis Reductor for parabolic equations.

    This reductor uses :class:`~pymor.reductors.basic.InstationaryRBReductor` for the actual
    RB-projection. The only addition is the assembly of an error estimator which
    bounds the discrete l2-in time / energy-in space error similar to [GP05]_, [HO08]_
    as follows:

    .. math::
        \left[ C_a^{-1}(\mu)\|e_N(\mu)\|^2 + \sum_{n=1}^{N} \Delta t\|e_n(\mu)\|^2_e \right]^{1/2}
            \leq \left[ C_a^{-1}(\mu)\Delta t \sum_{n=1}^{N}\|\mathcal{R}^n(u_n(\mu), \mu)\|^2_{e,-1}
                        + C_a^{-1}(\mu)\|e_0\|^2 \right]^{1/2}

    Here, :math:`\|\cdot\|` denotes the norm induced by the problem's mass matrix
    (e.g. the L^2-norm) and :math:`\|\cdot\|_e` is an arbitrary energy norm w.r.t.
    which the space operator :math:`A(\mu)` is coercive, and :math:`C_a(\mu)` is a
    lower bound for its coercivity constant. Finally, :math:`\mathcal{R}^n` denotes
    the implicit Euler timestepping residual for the (fixed) time step size :math:`\Delta t`,

    .. math::
        \mathcal{R}^n(u_n(\mu), \mu) :=
            f - M \frac{u_{n}(\mu) - u_{n-1}(\mu)}{\Delta t} - A(u_n(\mu), \mu),

    where :math:`M` denotes the mass operator and :math:`f` the source term.
    The dual norm of the residual is computed using the numerically stable projection
    from [BEOR14]_.

    Parameters
    ----------
    fom
        The |InstationaryModel| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    product
        The energy inner product |Operator| w.r.t. which the reduction error is
        estimated and `RB` is orthonormalized.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound :math:`C_a(\mu)`
        for the coercivity constant of `fom.operator` w.r.t. `product`.
    """

    def __init__(
        self,
        fom,
        RB=None,
        product=None,
        coercivity_estimator=None,
        check_orthonormality=None,
        check_tol=None,
    ):
        assert isinstance(fom.time_stepper, ImplicitEulerTimeStepper)
        super().__init__(
            fom,
            RB,
            product=product,
            check_orthonormality=check_orthonormality,
            check_tol=check_tol,
        )
        self.coercivity_estimator = coercivity_estimator

        self.residual_reductor = ImplicitEulerResidualReductor(
            self.bases["RB"],
            fom.operator,
            fom.mass,
            fom.T / fom.time_stepper.nt,
            rhs=fom.rhs,
            product=product,
        )

        self.initial_residual_reductor = ResidualReductor(
            self.bases["RB"],
            IdentityOperator(fom.solution_space),
            fom.initial_data,
            product=fom.l2_product,
            riesz_representatives=False,
        )

    def assemble_estimator(self):
        residual = self.residual_reductor.reduce()
        initial_residual = self.initial_residual_reductor.reduce()

        estimator = ParabolicRBEstimator(
            residual,
            self.residual_reductor.residual_range_dims,
            initial_residual,
            self.initial_residual_reductor.residual_range_dims,
            self.coercivity_estimator,
        )
        return estimator

    def assemble_estimator_for_subbasis(self, dims):
        return self._last_rom.estimator.restricted_to_subbasis(
            dims["RB"], m=self._last_rom
        )


class ParabolicRBEstimator(ImmutableInterface):
    """Instantiated by :class:`ParabolicRBReductor`.

    Not to be used directly.
    """

    def __init__(
        self,
        residual,
        residual_range_dims,
        initial_residual,
        initial_residual_range_dims,
        coercivity_estimator,
    ):
        self.residual = residual
        self.residual_range_dims = residual_range_dims
        self.initial_residual = initial_residual
        self.initial_residual_range_dims = initial_residual_range_dims
        self.coercivity_estimator = coercivity_estimator

    def estimate(self, U, mu, m, return_error_sequence=False):
        dt = m.T / m.time_stepper.nt
        C = self.coercivity_estimator(mu) if self.coercivity_estimator else 1.0

        est = np.empty(len(U))
        est[0] = (1.0 / C) * self.initial_residual.apply(U[0], mu=mu).l2_norm2()[0]
        est[1:] = self.residual.apply(
            U[1 : len(U)], U[0 : len(U) - 1], mu=mu
        ).l2_norm2()
        est[1:] *= dt / C ** 2
        est = np.sqrt(np.cumsum(est))

        return est if return_error_sequence else est[-1]

    def restricted_to_subbasis(self, dim, m):
        if self.residual_range_dims and self.initial_residual_range_dims:
            residual_range_dims = self.residual_range_dims[: dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            initial_residual_range_dims = self.initial_residual_range_dims[: dim + 1]
            initial_residual = self.initial_residual.projected_to_subbasis(
                initial_residual_range_dims[-1], dim
            )
            return ParabolicRBEstimator(
                residual,
                residual_range_dims,
                initial_residual,
                initial_residual_range_dims,
                self.coercivity_estimator,
            )
        else:
            self.logger.warning("Cannot efficiently reduce to subbasis")
            return ParabolicRBEstimator(
                self.residual.projected_to_subbasis(None, dim),
                None,
                self.initial_residual.projected_to_subbasis(None, dim),
                None,
                self.coercivity_estimator,
            )

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.image import estimate_image
from pymor.algorithms.projection import project
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.core.base import ImmutableObject
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import InstationaryRBReductor
from pymor.reductors.residual import ResidualReductor, ImplicitEulerResidualReductor


class ParabolicRBReductor(InstationaryRBReductor):
    r"""Reduced Basis Reductor for parabolic equations.

    This reductor uses :class:`~pymor.reductors.basic.InstationaryRBReductor` for the actual
    RB-projection. The only addition is the assembly of an error estimator which
    bounds the discrete l2-in time / energy-in space error similar to :cite:`GP05`, :cite:`HO08`
    as follows:

    .. math::
        \left[ C_a^{-1}(\mu)\|e_N(\mu)\|^2 + \sum_{n=1}^{N} \Delta t\|e_n(\mu)\|^2_e \right]^{1/2}
            \le \left[ C_a^{-2}(\mu)\Delta t \sum_{n=1}^{N}\|\mathcal{R}^n(u_n(\mu), \mu)\|^2_{e,-1}
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
    from :cite:`BEOR14`.

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

    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None):
        if not isinstance(fom.time_stepper, ImplicitEulerTimeStepper):
            raise NotImplementedError
        if fom.mass is not None and fom.mass.parametric and 't' in fom.mass.parameters:
            raise NotImplementedError
        super().__init__(fom, RB, product=product,
                         check_orthonormality=check_orthonormality, check_tol=check_tol)
        self.coercivity_estimator = coercivity_estimator

        self.residual_reductor = ImplicitEulerResidualReductor(
            self.bases['RB'],
            fom.operator,
            fom.mass,
            fom.T / fom.time_stepper.nt,
            rhs=fom.rhs,
            product=product
        )

        self.initial_residual_reductor = ResidualReductor(
            self.bases['RB'],
            IdentityOperator(fom.solution_space),
            fom.initial_data,
            product=fom.l2_product,
            riesz_representatives=False
        )

    def assemble_error_estimator(self):
        residual = self.residual_reductor.reduce()
        initial_residual = self.initial_residual_reductor.reduce()

        if self.fom.output_functional.linear:
            output_adjoint = self.fom.output_functional.H
            output_adjoint_riesz_range = estimate_image(vectors=(output_adjoint,), orthonormalize=True,
                                                        product=self.products['RB'], riesz_representatives=True)
            projected_output_adjoint = project(output_adjoint, output_adjoint_riesz_range, None)
        else:
            projected_output_adjoint = None

        return ParabolicRBEstimator(residual, self.residual_reductor.residual_range_dims, initial_residual,
                                    self.initial_residual_reductor.residual_range_dims, self.coercivity_estimator,
                                    projected_output_adjoint)

    def assemble_error_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class ParabolicRBEstimator(ImmutableObject):
    """Instantiated by :class:`ParabolicRBReductor`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims, initial_residual, initial_residual_range_dims,
                 coercivity_estimator, projected_output_adjoint=None):
        self.__auto_init(locals())

    def estimate_error(self, U, mu, m, return_error_sequence=False):
        dt = m.T / m.time_stepper.nt
        C = self.coercivity_estimator(mu) if self.coercivity_estimator else 1.

        est = np.empty(len(U))
        est[0] = (1./C) * self.initial_residual.apply(U[0], mu=mu).norm2()[0]
        if 't' in self.residual.parameters:
            t = 0
            for n in range(1, m.time_stepper.nt + 1):
                t += dt
                mu = mu.with_(t=t)
                est[n] = self.residual.apply(U[n], U[n-1], mu=mu).norm2()
        else:
            est[1:] = self.residual.apply(U[1:], U[:-1], mu=mu).norm2()
        est[1:] *= (dt/C**2)
        est = np.sqrt(np.cumsum(est))

        return est if return_error_sequence else est[-1]

    def estimate_output_error(self, U, mu, m, return_error_sequence=False, return_vector=False):
        if self.projected_output_adjoint is None:
            raise NotImplementedError
        estimate = self.estimate_error(U, mu, m, return_error_sequence=return_error_sequence)
        # scale with dual norm of the output functional
        output_functional_norms = self.projected_output_adjoint.as_range_array(mu).norm()
        errs = (estimate * output_functional_norms[:, np.newaxis]).T
        if return_vector:
            return errs
        else:
            return np.linalg.norm(errs, axis=1)

    def restricted_to_subbasis(self, dim, m):
        if self.residual_range_dims and self.initial_residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            initial_residual_range_dims = self.initial_residual_range_dims[:dim + 1]
            initial_residual = self.initial_residual.projected_to_subbasis(initial_residual_range_dims[-1], dim)
            return ParabolicRBEstimator(residual, residual_range_dims,
                                        initial_residual, initial_residual_range_dims,
                                        self.coercivity_estimator, self.projected_output_adjoint)
        else:
            self.logger.warning('Cannot efficiently reduce to subbasis')
            return ParabolicRBEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                        self.initial_residual.projected_to_subbasis(None, dim), None,
                                        self.coercivity_estimator, self.projected_output_adjoint)

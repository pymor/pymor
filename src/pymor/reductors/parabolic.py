# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.reductors.basic import GenericRBReductor
from pymor.reductors.residual import ResidualReductor, ImplicitEulerResidualReductor
from pymor.operators.constructions import IdentityOperator
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper


class ParabolicRBReductor(GenericRBReductor):
    r"""Reduced Basis Reductor for parabolic equations.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
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

    .. warning::
        The reduced basis `RB` is required to be orthonormal w.r.t. the given
        energy product. If not, the projection of the initial values will be
        computed incorrectly.

    .. [GP05]   M. A. Grepl, A. T. Patera, A Posteriori Error Bounds For Reduced-Basis
                Approximations Of Parametrized Parabolic Partial Differential Equations,
                M2AN 39(1), 157-181, 2005.
    .. [HO08]   B. Haasdonk, M. Ohlberger, Reduced basis method for finite volume
                approximations of parametrized evolution equations,
                M2AN 42(2), 277-302, 2008.

    Parameters
    ----------
    discretization
        The |InstationaryDiscretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    product
        The energy inner product |Operator| w.r.t. the reduction error is estimated.
        RB must be to be orthonomrmal w.r.t. this product!
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound :math:`C_a(\mu)`
        for the coercivity constant of `discretization.operator` w.r.t. `product`.
    extends
        Set by :meth:`~pymor.algorithms.greedy.greedy` to the result of the
        last reduction in case the basis extension was `hierarchic` (used to prevent
        re-computation of residual range basis vectors already obtained from previous
        reductions).

    Returns
    -------
    rd
        The reduced |Discretization|.
    rc
        The reconstructor providing a `reconstruct(U)` method which reconstructs
        high-dimensional solutions from solutions `U` of the reduced |Discretization|.
    reduction_data
        Additional data produced by the reduction process (compare the
        `extends` parameter).
    """
    def __init__(self, d, RB=None, product=None, coercivity_estimator=None):
        assert isinstance(d.time_stepper, ImplicitEulerTimeStepper)
        super().__init__(d, RB, product=product)
        self.coercivity_estimator = coercivity_estimator

        self.residual_reductor = ImplicitEulerResidualReductor(
            self.RB,
            d.operator,
            d.mass,
            d.T / d.time_stepper.nt,
            functional=d.rhs,
            product=product
        )

        self.initial_residual_reductor = ResidualReductor(
            self.RB,
            IdentityOperator(d.solution_space),
            d.initial_data,
            product=d.l2_product
        )

    def reduce(self):
        with self.logger.block('RB projection ...'):
            rd = super().reduce()

        with self.logger.block('Assembling error estimator ...'):
            residual = self.residual_reductor.reduce()
            initial_residual = self.initial_residual_reductor.reduce()

            estimator = ParabolicRBEstimator(residual, self.residual_reductor.residual_range_dims,
                                             initial_residual, self.initial_residual_reductor.residual_range_dims,
                                             self.coercivity_estimator)
            rd = rd.with_(estimator=estimator)

        return rd


class ParabolicRBEstimator(ImmutableInterface):
    """Instantiated by :func:`reduce_parabolic`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims, initial_residual, initial_residual_range_dims,
                 coercivity_estimator):
        self.residual = residual
        self.residual_range_dims = residual_range_dims
        self.initial_residual = initial_residual
        self.initial_residual_range_dims = initial_residual_range_dims
        self.coercivity_estimator = coercivity_estimator

    def estimate(self, U, mu, discretization, return_error_sequence=False):
        dt = discretization.T / discretization.time_stepper.nt
        C = self.coercivity_estimator(mu) if self.coercivity_estimator else 1.

        est = np.empty(len(U))
        est[0] = (1./C) * self.initial_residual.apply(U[0], mu=mu).l2_norm2()[0]
        est[1:] = self.residual.apply(U[1:len(U)], U[0:len(U)-1],
                                      mu=mu).l2_norm2()
        est[1:] *= (dt/C**2)
        est = np.sqrt(np.cumsum(est))

        return est if return_error_sequence else est[-1]

    def restricted_to_subbasis(self, dim, discretization):
        if self.residual_range_dims and self.initial_residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            initial_residual_range_dims = self.initial_residual_range_dims[:dim + 1]
            initial_residual = self.initial_residual.projected_to_subbasis(initial_residual_range_dims[-1], dim)
            return ParabolicRBEstimator(residual, residual_range_dims,
                                        initial_residual, initial_residual_range_dims,
                                        self.coercivity_estimator)
        else:
            self.logger.warn('Cannot efficiently reduce to subbasis')
            return ParabolicRBEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                        self.initial_residual.projected_to_subbasis(None, dim), None,
                                        self.coercivity_estimator)

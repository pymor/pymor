# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import ImmutableInterface
from pymor.reductors.basic import reduce_generic_rb
from pymor.reductors.residual import reduce_residual


def reduce_stationary_coercive(discretization, RB, error_product=None, coercivity_estimator=None,
                               disable_caching=True, extends=None):
    """Reductor for |StationaryDiscretizations| with coercive operator.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
    RB-projection. The only addition is an error estimator. The estimator evaluates the
    dual norm of the residual with respect to a given inner product. We use
    :func:`~pymor.reductors.residual.reduce_residual` for improved numerical stability.
    (See "A. Buhr, C. Engwer, M. Ohlberger, S. Rave, A Numerically Stable A Posteriori
    Error Estimator for Reduced Basis Approximations of Elliptic Equations,
    Proceedings of the 11th World Congress on Computational Mechanics, 2014.")

    Parameters
    ----------
    discretization
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    error_product
        Scalar product |Operator| used to calculate Riesz representative of the
        residual. If `None`, the Euclidean product is used.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound for the coercivity
        constant of the given problem. Note that the computed error estimate is only
        guaranteed to be an upper bound for the error when an appropriate coercivity
        estimate is specified.
    disable_caching
        If `True`, caching of solutions is disabled for the reduced |Discretization|.
    extends
        Set by :meth:`~pymor.algorithms.greedy.greedy` to the result of the
        last reduction in case the basis extension was `hierarchic`. Used to prevent
        re-computation of residual range basis vectors already obtained from previous
        reductions.

    Returns
    -------
    rd
        The reduced |Discretization|.
    rc
        The reconstructor providing a `reconstruct(U)` method which reconstructs
        high-dimensional solutions from solutions `U` of the reduced |Discretization|.
    reduction_data
        Additional data produced by the reduction process. (Compare the `extends`
        parameter.)
    """

    assert extends is None or len(extends) == 3

    old_residual_data = extends[2].pop('residual') if extends else None

    rd, rc, data = reduce_generic_rb(discretization, RB, disable_caching=disable_caching, extends=extends)

    residual, residual_reconstructor, residual_data = reduce_residual(discretization.operator, discretization.rhs, RB,
                                                                      product=error_product, extends=old_residual_data)

    estimator = StationaryCoerciveEstimator(residual, residual_data.get('residual_range_dims', None),
                                            coercivity_estimator)

    rd = rd.with_(estimator=estimator)

    data.update(residual=(residual, residual_reconstructor, residual_data))

    return rd, rc, data


class StationaryCoerciveEstimator(ImmutableInterface):
    """Instatiated by :meth:`reduce_stationary_coercive`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims, coercivity_estimator):
        self.residual = residual
        self.residual_range_dims = residual_range_dims
        self.coercivity_estimator = coercivity_estimator

    def estimate(self, U, mu, discretization):
        est = self.residual.apply(U, mu=mu).l2_norm()
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)
        return est

    def restricted_to_subbasis(self, dim, discretization):
        if self.residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            return StationaryCoerciveEstimator(residual, residual_range_dims, self.coercivity_estimator)
        else:
            self.logger.warn('Cannot efficiently reduce to subbasis')
            return StationaryCoerciveEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                               self.coercivity_estimator)

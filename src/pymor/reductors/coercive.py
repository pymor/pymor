# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.logger import getLogger
from pymor.core.interfaces import ImmutableInterface
from pymor.operators.constructions import LincombOperator, induced_norm
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import reduce_generic_rb
from pymor.reductors.residual import reduce_residual
from pymor.vectorarrays.numpy import NumpyVectorArray


def reduce_coercive(discretization, RB, product=None, coercivity_estimator=None,
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
    product
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

    logger = getLogger('pymor.reductors.coercive.reduce_coercive')

    old_residual_data = extends[2].pop('residual') if extends else None

    with logger.block('RB projection ...'):
        rd, rc, data = reduce_generic_rb(discretization, RB, disable_caching=disable_caching, extends=extends)

    with logger.block('Assembling error estimator ...'):
        residual, residual_reconstructor, residual_data \
            = reduce_residual(discretization.operator, discretization.rhs, RB,
                              product=product, extends=old_residual_data)

    estimator = ReduceCoerciveEstimator(residual, residual_data.get('residual_range_dims', None), coercivity_estimator)

    rd = rd.with_(estimator=estimator)

    data.update(residual=(residual, residual_reconstructor, residual_data))

    return rd, rc, data


class ReduceCoerciveEstimator(ImmutableInterface):
    """Instatiated by :meth:`reduce_coercive`.

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
            return ReduceCoerciveEstimator(residual, residual_range_dims, self.coercivity_estimator)
        else:
            self.logger.warn('Cannot efficiently reduce to subbasis')
            return ReduceCoerciveEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                           self.coercivity_estimator)


def reduce_coercive_simple(discretization, RB, product=None, coercivity_estimator=None,
                           disable_caching=True, extends=None):
    """Reductor for linear |StationaryDiscretizations| with affinely decomposed operator and rhs.

    .. note::
       The reductor :func:`reduce_coercive` can be used for arbitrary coercive
       |StationaryDiscretizations| and offers an improved error estimator
       with better numerical stability.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
    RB-projection. The only addition is an error estimator. The estimator evaluates the
    norm of the residual with respect to a given inner product.

    Parameters
    ----------
    discretization
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    product
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
        re-computation of Riesz representatives already obtained from previous
        reductions.

    Returns
    -------
    rd
        The reduced |Discretization|.
    rc
        The reconstructor providing a `reconstruct(U)` method which reconstructs
        high-dimensional solutions from solutions `U` of the reduced |Discretization|.
    reduction_data
        Additional data produced by the reduction process. In this case the computed
        Riesz representatives. (Compare the `extends` parameter.)
    """

    # assert isinstance(discretization, StationaryDiscretization)
    assert discretization.linear
    assert isinstance(discretization.operator, LincombOperator)
    assert all(not op.parametric for op in discretization.operator.operators)
    if discretization.rhs.parametric:
        assert isinstance(discretization.rhs, LincombOperator)
        assert all(not op.parametric for op in discretization.rhs.operators)
    assert extends is None or len(extends) == 3

    d = discretization
    rd, rc, data = reduce_generic_rb(d, RB, disable_caching=disable_caching, extends=extends)
    if extends:
        old_data = extends[2]
        old_RB_size = len(extends[1].RB)
    else:
        old_RB_size = 0

    # compute data for estimator
    space = d.operator.source

    # compute the Riesz representative of (U, .)_L2 with respect to product
    def riesz_representative(U):
        if product is None:
            return U.copy()
        else:
            return product.apply_inverse(U)

    def append_vector(U, R, RR):
        RR.append(riesz_representative(U), remove_from_other=True)
        R.append(U, remove_from_other=True)

    # compute all components of the residual
    if RB is None:
        RB = discretization.solution_space.empty()

    if extends:
        R_R, RR_R = old_data['R_R'], old_data['RR_R']
    elif not d.rhs.parametric:
        R_R = space.empty(reserve=1)
        RR_R = space.empty(reserve=1)
        append_vector(d.rhs.as_vector(), R_R, RR_R)
    else:
        R_R = space.empty(reserve=len(d.rhs.operators))
        RR_R = space.empty(reserve=len(d.rhs.operators))
        for op in d.rhs.operators:
            append_vector(op.as_vector(), R_R, RR_R)

    if len(RB) == 0:
        R_Os = [space.empty()]
        RR_Os = [space.empty()]
    elif not d.operator.parametric:
        R_Os = [space.empty(reserve=len(RB))]
        RR_Os = [space.empty(reserve=len(RB))]
        for i in range(len(RB)):
            append_vector(-d.operator.apply(RB, ind=i), R_Os[0], RR_Os[0])
    else:
        R_Os = [space.empty(reserve=len(RB)) for _ in range(len(d.operator.operators))]
        RR_Os = [space.empty(reserve=len(RB)) for _ in range(len(d.operator.operators))]
        if old_RB_size > 0:
            for op, R_O, RR_O, old_R_O, old_RR_O in zip(d.operator.operators, R_Os, RR_Os,
                                                         old_data['R_Os'], old_data['RR_Os']):
                R_O.append(old_R_O)
                RR_O.append(old_RR_O)
        for op, R_O, RR_O in zip(d.operator.operators, R_Os, RR_Os):
            for i in range(old_RB_size, len(RB)):
                append_vector(-op.apply(RB, [i]), R_O, RR_O)

    # compute Gram matrix of the residuals
    R_RR = RR_R.dot(R_R)
    R_RO = np.hstack([RR_R.dot(R_O) for R_O in R_Os])
    R_OO = np.vstack([np.hstack([RR_O.dot(R_O) for R_O in R_Os]) for RR_O in RR_Os])

    estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
    estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR
    estimator_matrix[len(R_RR):, len(R_RR):] = R_OO
    estimator_matrix[:len(R_RR), len(R_RR):] = R_RO
    estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T

    estimator_matrix = NumpyMatrixOperator(estimator_matrix)

    estimator = ReduceCoerciveSimpleEstimator(estimator_matrix, coercivity_estimator)
    rd = rd.with_(estimator=estimator)
    data.update(R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os)

    return rd, rc, data


class ReduceCoerciveSimpleEstimator(ImmutableInterface):
    """Instatiated by :meth:`reduce_coercive_simple`.

    Not to be used directly.
    """

    def __init__(self, estimator_matrix, coercivity_estimator):
        self.estimator_matrix = estimator_matrix
        self.coercivity_estimator = coercivity_estimator
        self.norm = induced_norm(estimator_matrix)

    def estimate(self, U, mu, discretization):
        d = discretization
        if len(U) > 1:
            raise NotImplementedError
        if not d.rhs.parametric:
            CR = np.ones(1)
        else:
            CR = np.array(d.rhs.evaluate_coefficients(mu))

        if not d.operator.parametric:
            CO = np.ones(1)
        else:
            CO = np.array(d.operator.evaluate_coefficients(mu))

        C = np.hstack((CR, np.dot(CO[..., np.newaxis], U.data).ravel()))

        est = self.norm(NumpyVectorArray(C))
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)

        return est

    def restricted_to_subbasis(self, dim, discretization):
        d = discretization
        cr = 1 if not d.rhs.parametric else len(d.rhs.operators)
        co = 1 if not d.operator.parametric else len(d.operator.operators)
        old_dim = d.operator.source.dim

        indices = np.concatenate((np.arange(cr),
                                 ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel() + cr))
        matrix = self.estimator_matrix._matrix[indices, :][:, indices]

        return ReduceCoerciveSimpleEstimator(NumpyMatrixOperator(matrix), self.coercivity_estimator)

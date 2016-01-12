# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.operators.constructions import LincombOperator, induced_norm
from pymor.reductors.basic import reduce_generic_rb
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray


def reduce_stationary_affine_linear(discretization, RB, error_product=None, coercivity_estimator=None,
                                    disable_caching=True, extends=None):
    """Reductor for linear |StationaryDiscretizations| with affinely decomposed operator and rhs.

    .. note::
       The reductor :func:`~pymor.reductors.stationary.reduce_stationary_coercive` can be used
       for arbitrary coercive |StationaryDiscretizations| and offers an improved error
       estimator with better numerical stability.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
    RB-projection. The only addition is an error estimator. The estimator evaluates the
    norm of the residual with respect to a given inner product.

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

    # compute the Riesz representative of (U, .)_L2 with respect to error_product
    def riesz_representative(U):
        if error_product is None:
            return U.copy()
        else:
            return error_product.apply_inverse(U)

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
        for i in xrange(len(RB)):
            append_vector(-d.operator.apply(RB, ind=i), R_Os[0], RR_Os[0])
    else:
        R_Os = [space.empty(reserve=len(RB)) for _ in xrange(len(d.operator.operators))]
        RR_Os = [space.empty(reserve=len(RB)) for _ in xrange(len(d.operator.operators))]
        if old_RB_size > 0:
            for op, R_O, RR_O, old_R_O, old_RR_O in izip(d.operator.operators, R_Os, RR_Os,
                                                         old_data['R_Os'], old_data['RR_Os']):
                R_O.append(old_R_O)
                RR_O.append(old_RR_O)
        for op, R_O, RR_O in izip(d.operator.operators, R_Os, RR_Os):
            for i in xrange(old_RB_size, len(RB)):
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

    estimator = StationaryAffineLinearReducedEstimator(estimator_matrix, coercivity_estimator)
    rd = rd.with_(estimator=estimator)
    data.update(R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os)

    return rd, rc, data


class StationaryAffineLinearReducedEstimator(ImmutableInterface):
    """Instatiated by :meth:`reduce_stationary_affine_linear`.

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

        return StationaryAffineLinearReducedEstimator(NumpyMatrixOperator(matrix), self.coercivity_estimator)

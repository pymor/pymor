# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import types

import numpy as np

from pymor.operators import LincombOperatorInterface, NumpyMatrixOperator
from pymor.core import BasicInterface
from pymor.discretizations import StationaryDiscretization
from pymor.la import NumpyVectorArray, induced_norm
from pymor.reductors.basic import reduce_generic_rb


def reduce_stationary_affine_linear(discretization, RB, error_product=None, disable_caching=True):
    '''Reductor for stationary linear problems whose `operator` and `rhs` are affinely decomposed.

    We simply use reduce_generic_rb for the actual RB-projection. The only addition
    is an error estimator. The estimator evaluates the norm of the residual with
    respect to a given inner product. We do not estimate the norm or the coercivity
    constant of the operator, therefore the estimated error can be lower than the
    actual error.

    Parameters
    ----------
    discretization
        The discretization which is to be reduced.
    RB
        The reduced basis (i.e. an array of vectors) on which to project.
    error_product
        Scalar product corresponding to the norm of the error. Used to calculate
        Riesz respresentatives of the components of the residual. If `None`, the
        standard L2-product is used.
    disable_caching
        If `True`, caching of the solutions of the reduced discretization
        is disabled.

    Returns
    -------
    rd
        The reduced discretization.
    rc
        The reconstructor providing a `reconstruct(U)` method which reconstructs
        high-dimensional solutions from solutions U of the reduced discretization.
    '''

    #assert isinstance(discretization, StationaryDiscretization)
    assert discretization.linear
    assert isinstance(discretization.operator, LincombOperatorInterface)
    assert all(not op.parametric for op in discretization.operator.operators)
    if discretization.rhs.parametric:
        assert isinstance(discretization.rhs, LincombOperatorInterface)
        assert all(not op.parametric for op in discretization.rhs.operators)

    d = discretization
    rd, rc = reduce_generic_rb(d, RB, product=None, disable_caching=disable_caching)

    # compute data for estimator
    space_dim = d.operator.dim_source
    space_type = d.operator.type_source

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
    rl = 1 if not d.rhs.parametric else len(d.rhs.operators)
    ol = 1 if not d.operator.parametric else len(d.operator.operators)

    if RB is None:
        RB = NumpyVectorArray(np.zeros((0, next(d.operators.itervalues()).dim_source)))

    if not d.rhs.parametric:
        R_R = space_type.empty(space_dim, reserve=1)
        RR_R = space_type.empty(space_dim, reserve=1)
        append_vector(d.rhs.as_vector(), R_R, RR_R)
    else:
        R_R = space_type.empty(space_dim, reserve=len(d.rhs.operators))
        RR_R = space_type.empty(space_dim, reserve=len(d.rhs.operators))
        for op in d.rhs.operators:
            append_vector(op.as_vector(), R_R, RR_R)

    if len(RB) == 0:
        R_O = space_type.empty(space_dim)
        RR_O = space_type.empty(space_dim)
    elif not d.operator.parametric:
        for i in xrange(len(RB)):
            append_vector(d.operator.apply(RB, ind=i), R_O, RR_O)
        R_O = space_type.empty(space_dim, reserve=len(RB))
        RR_O = space_type.empty(space_dim, reserve=len(RB))
    else:
        R_O = space_type.empty(space_dim, reserve=len(d.operator.operators) * len(RB))
        RR_O = space_type.empty(space_dim, reserve=len(d.operator.operators) * len(RB))
        for op in d.operator.operators:
            for i in xrange(len(RB)):
                append_vector(-op.apply(RB, [i]), R_O, RR_O)

    # compute Gram matrix of the residuals
    R_RR = RR_R.dot(R_R, pairwise=False)
    R_RO = RR_R.dot(R_O, pairwise=False)
    R_OO = RR_O.dot(R_O, pairwise=False)

    estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
    estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR
    estimator_matrix[len(R_RR):, len(R_RR):] = R_OO
    estimator_matrix[:len(R_RR), len(R_RR):] = R_RO
    estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T

    estimator_matrix = NumpyMatrixOperator(estimator_matrix)

    estimator = StationaryAffineLinearReducedEstimator(estimator_matrix)
    rd = rd.with_(estimator=estimator)

    return rd, rc


class StationaryAffineLinearReducedEstimator(BasicInterface):

    def __init__(self, estimator_matrix):
        self.estimator_matrix = estimator_matrix
        self.lock()

    def estimate(self, U, mu, discretization):
        d = discretization
        assert len(U) == 1, 'Can estimate only one solution vector'
        if not d.rhs.parametric:
            CR = np.ones(1)
        else:
            CR = d.rhs.evaluate_coefficients(mu)

        if not d.operator.parametric:
            CO = np.ones(1)
        else:
            CO = d.operator.evaluate_coefficients(mu)

        C = np.hstack((CR, np.dot(CO[..., np.newaxis], U.data).ravel()))

        return induced_norm(self.estimator_matrix)(NumpyVectorArray(C))

    def restricted_to_subbasis(self, dim, discretization):
        d = discretization
        cr = 1 if not d.rhs.parametric else len(d.rhs.operators)
        co = 1 if not d.operator.parametric else len(d.operator.operators)
        old_dim = d.operator.dim_source

        indices = np.concatenate((np.arange(cr),
                                 ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel() + cr))
        matrix = self.estimator_matrix._matrix[indices, :][:, indices]

        return StationaryAffineLinearReducedEstimator(NumpyMatrixOperator(matrix))

# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import types

import numpy as np

from pymor.operators import LinearAffinelyDecomposedOperator, NumpyLinearOperator
from pymor.operators.solvers import solve_linear
from pymor.discretizations import StationaryLinearDiscretization
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

    #assert isinstance(discretization, StationaryLinearDiscretization)
    assert isinstance(discretization.operator, LinearAffinelyDecomposedOperator)
    assert all(not op.parametric for op in discretization.operator.operators)
    assert discretization.operator.operator_affine_part is None\
        or not discretization.operator.operator_affine_part.parametric
    if discretization.rhs.parametric:
        assert isinstance(discretization.rhs, LinearAffinelyDecomposedOperator)
        assert all(not op.parametric for op in discretization.rhs.operators)
        assert discretization.rhs.operator_affine_part is None or not discretization.rhs.operator_affine_part.parametric

    d = discretization
    rd, rc = reduce_generic_rb(d, RB, product=None, disable_caching=disable_caching)

    # compute data for estimator
    space_dim = d.operator.dim_source
    space_type = d.operator.type_source

    if error_product is not None:
        error_product = error_product.assemble()

    solver = d.solver if hasattr(d, 'solver') else solve_linear

    # compute the Riesz representative of (U, .)_L2 with respect to error_product
    def riesz_representative(U):
        if error_product is None:
            return U.copy()
        else:
            return solver(error_product, U)

    def append_vector(U, R, RR):
        RR.append(riesz_representative(U), remove_from_other=True)
        R.append(U, remove_from_other=True)


    # compute all components of the residual
    ra = 1 if not d.rhs.parametric or d.rhs.operator_affine_part is not None else 0
    rl = 0 if not d.rhs.parametric else len(d.rhs.operators)
    oa = 1 if not d.operator.parametric or d.operator.operator_affine_part is not None else 0
    ol = 0 if not d.operator.parametric else len(d.operator.operators)

    # if RB is None: RB = np.zeros((0, d.operator.dim_source))
    if RB is None:
        RB = NumpyVectorArray(np.zeros((0, next(d.operators.itervalues()).dim_source)))

    R_R = space_type.empty(space_dim, reserve=ra + rl)
    R_O = space_type.empty(space_dim, reserve=(oa + ol) * len(RB))
    RR_R = space_type.empty(space_dim, reserve=ra + rl)
    RR_O = space_type.empty(space_dim, reserve=(oa + ol) * len(RB))

    if not d.rhs.parametric:
        append_vector(d.rhs.assemble().as_vector_array(), R_R, RR_R)

    if d.rhs.parametric and d.rhs.operator_affine_part is not None:
        append_vector(d.rhs.operator_affine_part.assemble().as_vector_array(), R_R, RR_R)

    if d.rhs.parametric:
        for op in d.rhs.operators:
            append_vector(op.assemble().as_vector_array(), R_R, RR_R)

    if len(RB) > 0 and not d.operator.parametric:
        for i in xrange(len(RB)):
            append_vector(d.operator.apply(RB, ind=[i]), R_O, RR_O)

    if len(RB) > 0 and d.operator.parametric and d.operator.operator_affine_part is not None:
        for i in xrange(len(RB)):
            append_vector(d.operator.operator_affine_part.apply(RB, ind=[i]), R_O, RR_O)

    if len(RB) > 0 and d.operator.parametric:
        for op in d.operator.operators:
            for i in xrange(len(RB)):
                append_vector(-op.apply(RB, [i]), R_O, RR_O)

    # compute Gram matrix of the residuals
    R_RR = RR_R.prod(R_R, pairwise=False)
    R_RO = RR_R.prod(R_O, pairwise=False)
    R_OO = RR_O.prod(R_O, pairwise=False)

    estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
    estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR
    estimator_matrix[len(R_RR):, len(R_RR):] = R_OO
    estimator_matrix[:len(R_RR), len(R_RR):] = R_RO
    estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T

    rd.estimator_matrix = NumpyLinearOperator(estimator_matrix)

    # this is our estimator
    def estimate(self, U, mu=None):
        assert len(U) == 1, 'Can estimate only one solution vector'
        if not self.rhs.parametric or self.rhs.operator_affine_part is not None:
            CRA = np.ones(1)
        else:
            CRA = np.ones(0)

        if self.rhs.parametric:
            CRL = self.rhs.evaluate_coefficients(self.map_parameter(mu, 'rhs'))
        else:
            CRL = np.ones(0)

        CR = np.hstack((CRA, CRL))

        if not self.operator.parametric or self.operator.operator_affine_part is not None:
            COA = np.ones(1)
        else:
            COA = np.ones(0)

        if self.operator.parametric:
            COL = self.operator.evaluate_coefficients(self.map_parameter(mu, 'operator'))
        else:
            COL = np.ones(0)

        C = np.hstack((CR, np.dot(np.hstack((COA, COL))[..., np.newaxis], U.data).ravel()))

        return induced_norm(self.estimator_matrix)(NumpyVectorArray(C))

    rd.estimate = types.MethodType(estimate, rd)

    return rd, rc


def numpy_reduce_stationary_affine_linear(discretization, RB, error_product=None, disable_caching=True):
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
    product
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

    assert isinstance(discretization, StationaryLinearDiscretization)
    assert isinstance(discretization.operator, LinearAffinelyDecomposedOperator)
    assert all(not op.parametric for op in discretization.operator.operators)
    assert discretization.operator.operator_affine_part is None\
        or not discretization.operator.operator_affine_part.parametric
    if discretization.rhs.parametric:
        assert isinstance(discretization.rhs, LinearAffinelyDecomposedOperator)
        assert all(not op.parametric for op in discretization.rhs.operators)
        assert discretization.rhs.operator_affine_part is None or not discretization.rhs.operator_affine_part.parametric

    d = discretization
    rd, rc = reduce_generic_rb(d, RB, product=None, disable_caching=disable_caching)

    # compute data for estimator
    space_dim = d.operator.dim_source

    # compute the Riesz representative of (U, .)_L2 with respect to error_product
    def riesz_representative(U):
        if error_product is None:
            return U
        return d.solver(error_product.assemble(), NumpyLinearOperator(U)).data.ravel()

    # compute all components of the residual
    ra = 1 if not d.rhs.parametric or d.rhs.operator_affine_part is not None else 0
    rl = 0 if not d.rhs.parametric else len(d.rhs.operators)
    oa = 1 if not d.operator.parametric or d.operator.operator_affine_part is not None else 0
    ol = 0 if not d.operator.parametric else len(d.operator.operators)

    # if RB is None: RB = np.zeros((0, d.operator.dim_source))
    if RB is None:
        RB = NumpyVectorArray(np.zeros((0, next(d.operators.itervalues()).dim_source)))
    R_R = np.empty((ra + rl, space_dim))
    R_O = np.empty(((oa + ol) * len(RB), space_dim))
    RR_R = np.empty((ra + rl, space_dim))
    RR_O = np.empty(((oa + ol) * len(RB), space_dim))

    if not d.rhs.parametric:
        R_R[0] = d.rhs.assemble()._matrix.ravel()
        RR_R[0] = riesz_representative(R_R[0])

    if d.rhs.parametric and d.rhs.operator_affine_part is not None:
        R_R[0] = d.rhs.operator_affine_part.assemble()._matrix.ravel()
        RR_R[0] = riesz_representative(R_R[0])

    if d.rhs.parametric:
        R_R[ra:] = np.array([op.assemble()._matrix.ravel() for op in d.rhs.operators])
        RR_R[ra:] = np.array(map(riesz_representative, R_R[ra:]))

    if len(RB) > 0 and not d.operator.parametric:
        R_O[0:len(RB)] = d.operator.apply(RB).data
        RR_O[0:len(RB)] = np.array(map(riesz_representative, R_O[0:len(RB)]))

    if len(RB) > 0 and d.operator.parametric and d.operator.operator_affine_part is not None:
        R_O[0:len(RB)] = d.operator.operator_affine_part.apply(RB).data
        RR_O[0:len(RB)] = np.array(map(riesz_representative, R_O[0:len(RB)]))

    if len(RB) > 0 and d.operator.parametric:
        for i, op in enumerate(d.operator.operators):
            A = R_O[(oa + i) * len(RB): (oa + i + 1) * len(RB)]
            A[:] = -op.apply(RB).data
            RR_O[(oa + i) * len(RB): (oa + i + 1) * len(RB)] = np.array(map(riesz_representative, A))

    # compute Gram matrix of the residuals
    R_RR = np.dot(RR_R, R_R.T)
    R_RO = np.dot(RR_R, R_O.T)
    R_OO = np.dot(RR_O, R_O.T)

    estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
    estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR
    estimator_matrix[len(R_RR):, len(R_RR):] = R_OO
    estimator_matrix[:len(R_RR), len(R_RR):] = R_RO
    estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T

    rd.estimator_matrix = NumpyLinearOperator(estimator_matrix)

    # this is our estimator
    def estimate(self, U, mu=None):
        assert len(U) == 1, 'Can estimate only one solution vector'
        if not self.rhs.parametric or self.rhs.operator_affine_part is not None:
            CRA = np.ones(1)
        else:
            CRA = np.ones(0)

        if self.rhs.parametric:
            CRL = self.rhs.evaluate_coefficients(self.map_parameter(mu, 'rhs'))
        else:
            CRL = np.ones(0)

        CR = np.hstack((CRA, CRL))

        if not self.operator.parametric or self.operator.operator_affine_part is not None:
            COA = np.ones(1)
        else:
            COA = np.ones(0)

        if self.operator.parametric:
            COL = self.operator.evaluate_coefficients(self.map_parameter(mu, 'operator'))
        else:
            COL = np.ones(0)

        C = np.hstack((CR, np.dot(np.hstack((COA, COL))[..., np.newaxis], U.data).ravel()))

        return induced_norm(self.estimator_matrix)(NumpyVectorArray(C))

    rd.estimate = types.MethodType(estimate, rd)

    return rd, rc

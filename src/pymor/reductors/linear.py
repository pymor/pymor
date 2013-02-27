from __future__ import absolute_import, division, print_function

import types

import numpy as np

import pymor.core as core
from pymor.core.cache import Cachable, NO_CACHE_CONFIG
from pymor.discreteoperators import LinearAffinelyDecomposedOperator, project_operator
from pymor.discretizations import StationaryLinearDiscretization
from pymor.tools import float_cmp_all
from pymor.la import induced_norm
from .basic import GenericRBReductor


class StationaryAffineLinearReductor(GenericRBReductor):
    '''Reductor for stationary linear problems whose operator and rhs are affinely
    decomposed.

    We simply use GenericRBReductor for the actual RB-projection. The only addition
    is an error estimator. The estimator evaluates the norm of the residual with
    respect to a given inner product. We do not estimate the norm or the coercivity
    constant of the operator, therefore the estimated error can be lower than the
    actual error.
    '''

    def __init__(self, discretization, error_product=None, disable_caching=True):
        assert isinstance(discretization, StationaryLinearDiscretization)
        assert isinstance(discretization.operator, LinearAffinelyDecomposedOperator)
        assert all(not op.parametric for op in discretization.operator.operators)
        assert discretization.operator.operator_affine_part is None or not discretization.operator.operator_affine_part.parametric
        if discretization.rhs.parametric:
            assert isinstance(discretization.rhs, LinearAffinelyDecomposedOperator)
            assert all(not op.parametric for op in discretization.rhs.operators)
            assert discretization.rhs.operator_affine_part is None or not discretization.rhs.operator_affine_part.parametric

        super(StationaryAffineLinearReductor, self).__init__(discretization, product=None, disable_caching=disable_caching)
        self.error_product = error_product


    def reduce(self, RB):
        rd, rc = super(StationaryAffineLinearReductor, self).reduce(RB)

        # compute data for estimator
        d = self.discretization

        space_dim = d.operator.dim_source

        # compute the Riesz representative of (U, .)_L2 with respect to self.error_product
        def riesz_representative(U):
            if self.error_product is None:
                return U
            return d.solver(self.error_product.matrix(), U)

        # compute all components of the residual
        ra = 1 if not d.rhs.parametric or d.rhs.operator_affine_part is not None else 0
        rl = 0 if not d.rhs.parametric else len(d.rhs.operators)
        oa = 1 if not d.operator.parametric or d.operator.operator_affine_part is not None else 0
        ol = 0 if not d.operator.parametric else len(d.operator.operators)

        # if RB is None: RB = np.zeros((0, d.operator.dim_source))
        if RB is None: RB = np.zeros((0, next(d.operators.itervalues()).dim_source))
        R_R = np.empty((ra + rl, space_dim))
        R_O = np.empty(((oa + ol) * len(RB), space_dim))
        RR_R = np.empty((ra + rl, space_dim))
        RR_O = np.empty(((oa + ol) * len(RB), space_dim))

        if not d.rhs.parametric:
            R_R[0] = d.rhs.matrix().ravel()
            RR_R[0] = riesz_representative(R_R[0])

        if d.rhs.parametric and d.rhs.operator_affine_part is not None:
            R_R[0] = d.rhs.operator_affine_part.matrix().ravel()
            RR_R[0] = riesz_representative(R_R[0])

        if d.rhs.parametric:
            R_R[ra:] = np.array([op.matrix().ravel() for op in enumerate(d.rhs.operators)])
            RR_R[ra:] = np.array(map(riesz_representative, R_R[ra:]))

        if len(RB) > 0 and not d.operator.parametric:
            R_O[0:len(RB)] = np.array([d.operator.apply(B) for B in RB])
            RR_O[0:len(RB)] = np.array(map(riesz_representative, R_O[0:len(RB)]))

        if len(RB) > 0 and d.operator.parametric and d.operator.operator_affine_part is not None:
            R_O[0:len(RB)] = np.array([d.operator.operator_affine_part.apply(B) for B in RB])
            RR_O[0:len(RB)] = np.array(map(riesz_representative, R_O[0:len(RB)]))

        if len(RB) > 0 and d.operator.parametric:
            for i, op in enumerate(d.operator.operators):
                A = R_O[(oa + i) * len(RB): (oa + i + 1) * len(RB)]
                A[:] = np.array([- op.apply(B) for B in RB])
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

        rd.estimator_matrix = estimator_matrix

        # this is our estimator
        def estimate(self, U, mu={}):
            assert U.ndim == 1, 'Can estimate only one solution vector'
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

            C = np.hstack((CR, np.dot(np.hstack((COA, COL))[..., np.newaxis], U[np.newaxis, ...]).ravel()))

            return induced_norm(self.estimator_matrix)(C)

        rd.estimate = types.MethodType(estimate, rd)

        return rd, rc

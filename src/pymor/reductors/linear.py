from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.core.cache import Cachable, NO_CACHE_CONFIG
from pymor.discreteoperators import LinearAffinelyDecomposedOperator, project_operator
from pymor.discretizations import StationaryLinearDiscretization
from pymor.tools import float_cmp_all
from .basic import GenericRBReductor


class StationaryAffineLinearReductor(GenericRBReductor):

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

        d = self.discretization

        space_dim = d.operator.source_dim

        def riesz_representative(U):
            if self.error_product is None:
                return U
            return d.solver(self.error_product.matrix(), U)

        ra = 1 if d.rhs.parametric and d.rhs.operator_affine_part is not None else 0
        rl = 0 if not d.rhs.parametric else len(d.rhs.operators)
        oa = 1 if d.operator.parametric and d.operator.operator_affine_part is not None else 0
        ol = 0 if not d.operator.parametric else len(d.operator.operators)

        R_R = np.empty((ra + rl, space_dim))
        R_O = np.empty(((oa + ol) * len(RB), space_dim))
        RR_R = np.empty((ra + rl, space_dim))
        RR_O = np.empty(((oa + ol) * len(RB), space_dim))

        if d.rhs.parametric and d.rhs.operator_affine_part is not None:
            R_R[0] = d.rhs.operator_affine_part.matrix().ravel()
            RR_R[0] = riesz_representative(R_R[0])

        if d.rhs.parametric:
            R_R[ra:] = np.array([op.matrix().ravel() for op in enumerate(d.rhs.operators)])
            RR_R[ra:] = np.array(map(riesz_representative, R_R[ra:]))

        if len(RB) > 0 and d.operator.parametric and d.operator.operator_affine_part is not None:
            R_O[0:len(RB)] = np.array([d.operator.operator_affine_part.apply(B) for B in RB])
            RR_O[0:len(RB)] = np.array(map(riesz_representative, R_O[0:len(RB)]))

        if len(RB) > 0 and d.operator.parametric:
            for i, op in enumerate(d.operator.operators):
                A = R_O[(oa + i) * len(RB): (oa + i + 1) * len(RB)]
                A = np.array([- op.apply(B) for B in RB])
                R_O[(oa + i) * len(RB): (oa + i + 1) * len(RB)] = np.array(map(riesz_representative, A))

        rd.R_RR = np.dot(RR_R, R_R.T)
        rd.R_RO = np.dot(RR_R, R_O.T)
        rd.R_OO = np.dot(RR_O, R_O.T)

        return rd, rc

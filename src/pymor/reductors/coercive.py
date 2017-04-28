# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.operators.constructions import LincombOperator, induced_norm
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import GenericRBReductor
from pymor.reductors.residual import ResidualReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace


class CoerciveRBReductor(GenericRBReductor):
    """Reduced Basis reductor for |StationaryDiscretizations| with coercive linear operator.

    The only addition tor GenericRBReductor is an error estimator which evaluates the
    dual norm of the residual with respect to a given inner product. For the reduction
    of the residual we use :func:`~pymor.reductors.residual.reduce_residual` for
    improved numerical stability [BEOR14]_.

    .. [BEOR14] A. Buhr, C. Engwer, M. Ohlberger, S. Rave, A Numerically Stable A
                Posteriori Error Estimator for Reduced Basis Approximations of Elliptic
                Equations, Proceedings of the 11th World Congress on Computational
                Mechanics, 2014.

    Parameters
    ----------
    d
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the (initial) reduced basis on which to project.
    orthogonal_projection
        List of keys in `d.operators` for which the corresponding |Operator|
        should be orthogonally projected (i.e. operators which map to vectors in
        contrast to bilinear forms which map to functionals).
    product
        Inner product for the projection of the |Operators| given by
        `orthogonal_projection` and for the computation of Riesz representatives
        of the residual. If `None`, the Euclidean product is used.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound for the coercivity
        constant of the given problem. Note that the computed error estimate is only
        guaranteed to be an upper bound for the error when an appropriate coercivity
        estimate is specified.
    disable_caching
        If `True`, caching of solutions is disabled for the reduced |Discretization|.
    """

    def __init__(self, d, RB=None, orthogonal_projection=('initial_data',), product=None,
                 coercivity_estimator=None, disable_caching=True):
        super().__init__(d, RB,
                         orthogonal_projection=orthogonal_projection,
                         product=product,
                         disable_caching=disable_caching)
        self.coercivity_estimator = coercivity_estimator
        self.residual_reductor = ResidualReductor(self.RB, self.d.operator, self.d.rhs,
                                                  product=product)

    def reduce(self):
        with self.logger.block('RB projection ...'):
            rd = super().reduce()

        with self.logger.block('Assembling error estimator ...'):
            residual = self.residual_reductor.reduce()

            estimator = CoerciveRBEstimator(residual, self.residual_reductor.residual_range_dims.copy(),
                                            self.coercivity_estimator)
            rd = rd.with_(estimator=estimator)

        return rd


class CoerciveRBEstimator(ImmutableInterface):
    """Instantiated by :meth:`reduce_coercive`.

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
            return CoerciveRBEstimator(residual, residual_range_dims, self.coercivity_estimator)
        else:
            self.logger.warn('Cannot efficiently reduce to subbasis')
            return CoerciveRBEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                       self.coercivity_estimator)


class SimpleCoerciveRBReductor(GenericRBReductor):
    """Reductor for linear |StationaryDiscretizations| with affinely decomposed operator and rhs.

    .. note::
       The reductor :func:`reduce_coercive` can be used for arbitrary coercive
       |StationaryDiscretizations| and offers an improved error estimator
       with better numerical stability.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
    reduced basis projection. The only addition is an error estimator. The estimator
    evaluates the norm of the residual with respect to a given inner product.

    Parameters
    ----------
    d
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the (initial) reduced basis on which to project.
    orthogonal_projection
        List of keys in `d.operators` for which the corresponding |Operator|
        should be orthogonally projected (i.e. operators which map to vectors in
        contrast to bilinear forms which map to functionals).
    product
        Inner product for the projection of the |Operators| given by
        `orthogonal_projection` and for the computation of Riesz representatives
        of the residual. If `None`, the Euclidean product is used.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound for the coercivity
        constant of the given problem. Note that the computed error estimate is only
        guaranteed to be an upper bound for the error when an appropriate coercivity
        estimate is specified.
    disable_caching
        If `True`, caching of solutions is disabled for the reduced |Discretization|.
    """

    def __init__(self, d, RB=None, orthogonal_projection=('initial_data',), product=None,
                 coercivity_estimator=None, disable_caching=True):
        assert d.linear
        assert isinstance(d.operator, LincombOperator)
        assert all(not op.parametric for op in d.operator.operators)
        if d.rhs.parametric:
            assert isinstance(d.rhs, LincombOperator)
            assert all(not op.parametric for op in d.rhs.operators)

        super().__init__(d, RB,
                         orthogonal_projection=orthogonal_projection,
                         product=product,
                         disable_caching=disable_caching)
        self.coercivity_estimator = coercivity_estimator
        self.residual_reductor = ResidualReductor(self.RB, self.d.operator, self.d.rhs,
                                                  product=product)
        self.extends = None

    def reduce(self):
        d, RB, extends = self.d, self.RB, self.extends
        rd = super().reduce()
        if extends:
            old_RB_size = extends[0]
            old_data = extends[1]
        else:
            old_RB_size = 0

        # compute data for estimator
        space = d.operator.source

        # compute the Riesz representative of (U, .)_L2 with respect to product
        def riesz_representative(U):
            if self.product is None:
                return U.copy()
            else:
                return self.product.apply_inverse(U)

        def append_vector(U, R, RR):
            RR.append(riesz_representative(U), remove_from_other=True)
            R.append(U, remove_from_other=True)

        # compute all components of the residual
        if extends:
            R_R, RR_R = old_data['R_R'], old_data['RR_R']
        elif not d.rhs.parametric:
            R_R = space.empty(reserve=1)
            RR_R = space.empty(reserve=1)
            append_vector(d.rhs.as_source_array(), R_R, RR_R)
        else:
            R_R = space.empty(reserve=len(d.rhs.operators))
            RR_R = space.empty(reserve=len(d.rhs.operators))
            for op in d.rhs.operators:
                append_vector(op.as_source_array(), R_R, RR_R)

        if len(RB) == 0:
            R_Os = [space.empty()]
            RR_Os = [space.empty()]
        elif not d.operator.parametric:
            R_Os = [space.empty(reserve=len(RB))]
            RR_Os = [space.empty(reserve=len(RB))]
            for i in range(len(RB)):
                append_vector(-d.operator.apply(RB[i]), R_Os[0], RR_Os[0])
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
                    append_vector(-op.apply(RB[i]), R_O, RR_O)

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

        estimator = ReduceCoerciveSimpleEstimator(estimator_matrix, self.coercivity_estimator)
        rd = rd.with_(estimator=estimator)

        self.extends = (len(RB), dict(R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os))

        return rd


class ReduceCoerciveSimpleEstimator(ImmutableInterface):
    """Instantiated by :meth:`reduce_coercive_simple`.

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

        est = self.norm(NumpyVectorSpace.make_array(C))
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

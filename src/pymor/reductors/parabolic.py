# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.operators.constructions import LincombOperator, induced_norm
from pymor.reductors.basic import reduce_generic_rb
from pymor.reductors.residual import reduce_implicit_euler_residual
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper


def reduce_parabolic_l2_estimate(discretization, RB, disable_caching=True, extends=None):
    """Reductor for |InstationaryDiscretizations| with coercive operator.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
    RB-projection. The only addition is an error estimator. The estimator evaluates the
    dual norm of the residual with respect to the inner product induced through the mass operator.
    (See "B. Haasdonk, M. Ohlberger, Reduced basis method for finite volume approximations of parametrized evolution
    equations. M2AN 42(2), 277-302, 2008." and "M. A. Grepl, A. T. Patera, A Posteriori Error Bounds For Reduced-Basis
    Approximations Of Parametrized Parabolic Partial Differential Equations, M2AN 39(1), 157-181, 2005")
    We use :func:`~pymor.reductors.residual.reduce_implicit_euler_residual` for improved numerical stability.
    (See "A. Buhr, C. Engwer, M. Ohlberger, S. Rave, A Numerically Stable A Posteriori
    Error Estimator for Reduced Basis Approximations of Elliptic Equations,
    Proceedings of the 11th World Congress on Computational Mechanics, 2014.")

    Parameters
    ----------
    discretization
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
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
    assert isinstance(discretization.time_stepper, ImplicitEulerTimeStepper)

    old_residual_data = extends[2].pop('residual') if extends else None

    rd, rc, data = reduce_generic_rb(discretization, RB, disable_caching=disable_caching, extends=extends)

    dt = discretization.T / discretization.time_stepper.nt

    residual, residual_reconstructor, residual_data = reduce_implicit_euler_residual(discretization.operator,
                                                                                     discretization.mass,
                                                                                     dt,
                                                                                     discretization.rhs,
                                                                                     RB,
                                                                                     product=discretization.mass,
                                                                                     extends=old_residual_data)

    estimator = ReduceParabolicL2Estimator(residual, residual_data.get('residual_range_dims', None))

    rd = rd.with_(estimator=estimator)

    data.update(residual=(residual, residual_reconstructor, residual_data))

    return rd, rc, data


class ReduceParabolicL2Estimator(ImmutableInterface):
    """Instatiated by :meth:`reduce_parabolic_l2_estimate`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims):
        self.residual = residual
        self.residual_range_dims = residual_range_dims

    def estimate(self, U, mu, discretization, k=None, return_error_trajectory=False):
        est = [np.array([0.])]
        max_k = len(U) - 1 if k is None else k
        for i in xrange(max_k):
            est.append(est[i] + self.residual.apply(U, U, ind=i+1, ind_old=i, mu=mu).l2_norm())

        est = np.array(est)
        dt = discretization.T / discretization.time_stepper.nt
        est *= dt

        if not return_error_trajectory:
            return est[-1]
        else:
            return NumpyVectorArray(est)

    def restricted_to_subbasis(self, dim, discretization):
        if self.residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            return ReduceParabolicL2Estimator(residual, residual_range_dims)
        else:
            self.logger.warn('Cannot efficiently reduce to subbasis')
            return ReduceParabolicL2Estimator(self.residual.projected_to_subbasis(None, dim), None)


def reduce_parabolic_energy_estimate(discretization, RB, error_product=None, coercivity_estimator=None, gamma=0.,
                                     disable_caching=True, extends=None):
    """Reductor for |InstationaryDiscretizations| with coercive operator.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
    RB-projection. The only addition is an error estimator. The estimator evaluates the
    dual norm of the residual with respect to a given inner product.
    (See "B. Haasdonk, M. Ohlberger, Reduced basis method for finite volume approximations of parametrized evolution
    equations. M2AN 42(2), 277-302, 2008." and "M. A. Grepl, A. T. Patera, A Posteriori Error Bounds For Reduced-Basis
    Approximations Of Parametrized Parabolic Partial Differential Equations, M2AN 39(1), 157-181, 2005")
    We use :func:`~pymor.reductors.residual.reduce_implicit_euler_residual` for improved numerical stability.
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
    assert isinstance(discretization.time_stepper, ImplicitEulerTimeStepper)
    assert gamma is None or (0 <= gamma < 2)

    old_residual_data = extends[2].pop('residual') if extends else None

    rd, rc, data = reduce_generic_rb(discretization, RB, disable_caching=disable_caching, extends=extends)

    dt = discretization.T / discretization.time_stepper.nt

    residual, residual_reconstructor, residual_data = reduce_implicit_euler_residual(discretization.operator,
                                                                                     discretization.mass,
                                                                                     dt,
                                                                                     discretization.rhs,
                                                                                     RB,
                                                                                     product=error_product,
                                                                                     extends=old_residual_data)

    estimator = ReduceParabolicEnergyEstimator(residual, residual_data.get('residual_range_dims', None),
                                               coercivity_estimator, gamma)

    rd = rd.with_(estimator=estimator)

    data.update(residual=(residual, residual_reconstructor, residual_data))

    return rd, rc, data


class ReduceParabolicEnergyEstimator(ImmutableInterface):
    """Instatiated by :meth:`reduce_parabolic_energy_estimate`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims, coercivity_estimator, gamma):
        self.residual = residual
        self.residual_range_dims = residual_range_dims
        self.coercivity_estimator = coercivity_estimator
        self.gamma = gamma

    def estimate(self, U, mu, discretization, k=None, return_error_trajectory=False):
        est = [np.array([0.])]
        max_k = len(U) - 1 if k is None else k
        for i in xrange(max_k):
            est.append(est[i] + self.residual.apply(U, U, ind=i+1, ind_old=i, mu=mu).l2_norm()**2)

        est = np.array(est)
        dt = discretization.T / discretization.time_stepper.nt
        est *= dt

        est /= 2. - self.gamma

        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)

        if not return_error_trajectory:
            return np.sqrt(est[-1])
        else:
            return NumpyVectorArray(np.sqrt(est))

    def restricted_to_subbasis(self, dim, discretization):
        if self.residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            return ReduceParabolicEnergyEstimator(residual, residual_range_dims, self.coercivity_estimator, self.gamma)
        else:
            self.logger.warn('Cannot efficiently reduce to subbasis')
            return ReduceParabolicEnergyEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                                  self.coercivity_estimator, self.gamma)


def collect_residual_data(discretization, RB, error_product=None, extends=None):
    if extends:
        old_data = extends[2]
        old_RB_size = len(extends[1].RB)
    else:
        old_RB_size = 0

    # compute data for estimator
    space = discretization.operator.source

    # compute the Riesz representative of (U, .)_L2 with respect to err_product
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

    if len(RB) == 0:
        R_Ms = [space.empty()]
        RR_Ms = [space.empty()]
    else:
        R_Ms = [space.empty(reserve=len(RB))]
        RR_Ms = [space.empty(reserve=len(RB))]
        for i in xrange(len(RB)):
            append_vector(-discretization.mass.apply(RB, ind=i), R_Ms[0], RR_Ms[0])

    if extends:
        R_R, RR_R = old_data['R_R'], old_data['RR_R']
    elif not discretization.rhs.parametric:
        R_R = space.empty(reserve=1)
        RR_R = space.empty(reserve=1)
        append_vector(discretization.rhs.as_vector(), R_R, RR_R)
    else:
        R_R = space.empty(reserve=len(discretization.rhs.operators))
        RR_R = space.empty(reserve=len(discretization.rhs.operators))
        for op in discretization.rhs.operators:
            append_vector(op.as_vector(), R_R, RR_R)

    if len(RB) == 0:
        R_Os = [space.empty()]
        RR_Os = [space.empty()]
    elif not discretization.operator.parametric:
        R_Os = [space.empty(reserve=len(RB))]
        RR_Os = [space.empty(reserve=len(RB))]
        for i in xrange(len(RB)):
            append_vector(-d.operator.apply(RB, ind=i), R_Os[0], RR_Os[0])
    else:
        R_Os = [space.empty(reserve=len(RB)) for _ in xrange(len(discretization.operator.operators))]
        RR_Os = [space.empty(reserve=len(RB)) for _ in xrange(len(discretization.operator.operators))]
        if old_RB_size > 0:
            for op, R_O, RR_O, old_R_O, old_RR_O in izip(discretization.operator.operators, R_Os, RR_Os,
                                                         old_data['R_Os'], old_data['RR_Os']):
                R_O.append(old_R_O)
                RR_O.append(old_RR_O)
        for op, R_O, RR_O in izip(discretization.operator.operators, R_Os, RR_Os):
            for i in xrange(old_RB_size, len(RB)):
                append_vector(-op.apply(RB, [i]), R_O, RR_O)

    return R_Ms, RR_Ms, R_R, RR_R, R_Os, RR_Os


def compute_estimator_matrix(R_Ms, RR_Ms, R_R, RR_R, R_Os, RR_Os, dt):
    # compute Gram matrix of the residuals
    R_RR = RR_R.dot(R_R)
    R_RO = np.hstack([RR_R.dot(R_O) for R_O in R_Os])
    R_OO = np.vstack([np.hstack([RR_O.dot(R_O) for R_O in R_Os]) for RR_O in RR_Os])
    R_MM = np.vstack([np.hstack([RR_M.dot(R_M) for R_M in R_Ms]) for RR_M in RR_Ms])
    R_RM = np.hstack([RR_R.dot(R_M) for R_M in R_Ms])
    R_OM = np.vstack([np.hstack([RR_O.dot(R_M) for R_M in R_Ms]) for RR_O in RR_Os])

    estimator_matrix = np.empty((len(R_MM) + len(R_RR) + len(R_OO),) * 2)
    estimator_matrix[:len(R_MM), :len(R_MM)] = R_MM / (dt**2)
    estimator_matrix[len(R_MM):len(R_MM) + len(R_RR), len(R_MM):len(R_MM) + len(R_RR)] = R_RR
    estimator_matrix[len(R_MM) + len(R_RR):, len(R_MM) + len(R_RR):] = R_OO
    estimator_matrix[len(R_MM):len(R_MM) + len(R_RR), :len(R_MM)] = R_RM / dt
    estimator_matrix[:len(R_MM), len(R_MM):len(R_MM) + len(R_RR)] = R_RM.T / dt
    estimator_matrix[len(R_MM) + len(R_RR):, :len(R_MM)] = R_OM / dt
    estimator_matrix[:len(R_MM), len(R_MM) + len(R_RR):] = R_OM.T / dt
    estimator_matrix[len(R_MM):len(R_MM) + len(R_RR), len(R_MM) + len(R_RR):] = R_RO
    estimator_matrix[len(R_MM) + len(R_RR):, len(R_MM):len(R_MM) + len(R_RR)] = R_RO.T

    return NumpyMatrixOperator(estimator_matrix)


def reduce_parabolic_l2_estimate_simple(discretization, RB, disable_caching=True, extends=None):
    """Reductor for linear |InstationaryDiscretizations| with affinely decomposed operator and rhs.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
    RB-projection. The only addition is an error estimator. The estimator evaluates the
    norm of the residual with respect to the inner product induced through the mass operator.
    (See "B. Haasdonk, M. Ohlberger, Reduced basis method for finite volume approximations of parametrized evolution
    equations. M2AN 42(2), 277-302, 2008." and "M. A. Grepl, A. T. Patera, A Posteriori Error Bounds For Reduced-Basis
    Approximations Of Parametrized Parabolic Partial Differential Equations, M2AN 39(1), 157-181, 2005")

    Parameters
    ----------
    discretization
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
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

    assert discretization.linear
    assert isinstance(discretization.operator, LincombOperator)
    assert all(not op.parametric for op in discretization.operator.operators)
    if discretization.rhs.parametric:
        assert isinstance(discretization.rhs, LincombOperator)
        assert all(not op.parametric for op in discretization.rhs.operators)
    assert extends is None or len(extends) == 3
    assert isinstance(discretization.time_stepper, ImplicitEulerTimeStepper)

    rd, rc, data = reduce_generic_rb(discretization, RB, disable_caching=disable_caching, extends=extends)
    R_Ms, RR_Ms, R_R, RR_R, R_Os, RR_Os = collect_residual_data(discretization, RB, error_product=discretization.mass,
                                                                extends=extends)

    dt = discretization.T / discretization.time_stepper.nt
    estimator_matrix = compute_estimator_matrix(R_Ms, RR_Ms, R_R, RR_R, R_Os, RR_Os, dt)
    estimator = ReduceParabolicSimpleL2Estimator(estimator_matrix)

    rd = rd.with_(estimator=estimator)
    data.update(R_Ms=R_Ms, RR_Ms=RR_Ms, R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os)

    return rd, rc, data


class ReduceParabolicSimpleL2Estimator(ImmutableInterface):
    """Instatiated by :meth:`reduce_parabolic_l2_estimate_simple`.

    Not to be used directly.
    """

    def __init__(self, estimator_matrix):
        self.estimator_matrix = estimator_matrix
        self.norm = induced_norm(estimator_matrix, tol=1e-9)

    def estimate(self, U, mu, discretization, k=None, return_error_trajectory=False):
        d = discretization
        CM = np.ones(1)
        if not d.rhs.parametric:
            CR = np.ones(1)
        else:
            CR = np.array(d.rhs.evaluate_coefficients(mu))

        if not d.operator.parametric:
            CO = np.ones(1)
        else:
            CO = np.array(d.operator.evaluate_coefficients(mu))

        est = [np.array([0.])]
        max_k = len(U) - 1 if k is None else k
        for i in xrange(max_k):
            C = np.hstack((np.dot(CM[..., np.newaxis], (U.data[i + 1] - U.data[i])[np.newaxis, ...]).ravel(), CR,
                           np.dot(CO[..., np.newaxis], U.data[i + 1][np.newaxis, ...]).ravel()))

            est.append(est[i] + self.norm(NumpyVectorArray(C)))

        est = np.array(est)
        dt = d.T / discretization.time_stepper.nt
        est *= dt

        if not return_error_trajectory:
            return est[-1]
        else:
            return NumpyVectorArray(est)

    def restricted_to_subbasis(self, dim, discretization):
        d = discretization
        cr = 1 if not d.rhs.parametric else len(d.rhs.operators)
        co = 1 if not d.operator.parametric else len(d.operator.operators)
        old_dim = d.operator.source.dim

        indices = np.concatenate((np.arange(dim),
                                  np.arange(cr) + old_dim,
                                  ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel() + old_dim + cr))
        matrix = self.estimator_matrix._matrix[indices, :][:, indices]

        return ReduceParabolicSimpleL2Estimator(NumpyMatrixOperator(matrix))


def reduce_parabolic_energy_estimate_simple(discretization, RB, error_product=None, coercivity_estimator=None, gamma=0.,
                                        disable_caching=True, extends=None):
    """Reductor for linear |InstationaryDiscretizations| with affinely decomposed operator and rhs.

    This reductor uses :meth:`~pymor.reductors.basic.reduce_generic_rb` for the actual
    RB-projection. The only addition is an error estimator. The estimator evaluates the
    norm of the residual with respect to a given inner product.
    (See "B. Haasdonk, M. Ohlberger, Reduced basis method for finite volume approximations of parametrized evolution
    equations. M2AN 42(2), 277-302, 2008." and "M. A. Grepl, A. T. Patera, A Posteriori Error Bounds For Reduced-Basis
    Approximations Of Parametrized Parabolic Partial Differential Equations, M2AN 39(1), 157-181, 2005")

    Parameters
    ----------
    discretization
        The |Discretization| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    error_produt
        Scalar product |Operator| used to calculate Riesz representative of the
        residual. If `None`, the Euclidean product is used.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound for the coercivity
        constant of the given problem. Note that the computed error estimate is only
        guaranteed to be an upper bound for the error when an appropriate coercivity
        estimate is specified.
    gamma
        Value of gamma in the energy norm estimate.
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

    assert discretization.linear
    assert isinstance(discretization.operator, LincombOperator)
    assert all(not op.parametric for op in discretization.operator.operators)
    if discretization.rhs.parametric:
        assert isinstance(discretization.rhs, LincombOperator)
        assert all(not op.parametric for op in discretization.rhs.operators)
    assert extends is None or len(extends) == 3
    assert isinstance(discretization.time_stepper, ImplicitEulerTimeStepper)
    assert gamma is None or (0 <= gamma < 2)

    rd, rc, data = reduce_generic_rb(discretization, RB, disable_caching=disable_caching, extends=extends)

    R_Ms, RR_Ms, R_R, RR_R, R_Os, RR_Os = collect_residual_data(discretization, RB, error_product=error_product,
                                                                extends=extends)

    dt = discretization.T / discretization.time_stepper.nt
    estimator_matrix = compute_estimator_matrix(R_Ms, RR_Ms, R_R, RR_R, R_Os, RR_Os, dt)
    estimator = ReduceParabolicSimpleEnergyEstimator(estimator_matrix, coercivity_estimator, gamma)

    rd = rd.with_(estimator=estimator)
    data.update(R_Ms=R_Ms, RR_Ms=RR_Ms, R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os)

    return rd, rc, data


class ReduceParabolicSimpleEnergyEstimator(ImmutableInterface):
    """Instatiated by :meth:`reduce_parabolic_energy_estimate_simple`.

    Not to be used directly.
    """

    def __init__(self, estimator_matrix, coercivity_estimator, gamma):
        self.estimator_matrix = estimator_matrix
        self.coercivity_estimator = coercivity_estimator
        self.gamma = gamma

    def estimate(self, U, mu, discretization, k=None, return_error_trajectory=False):
        d = discretization
        CM = np.ones(1)
        if not d.rhs.parametric:
            CR = np.ones(1)
        else:
            CR = np.array(d.rhs.evaluate_coefficients(mu))

        if not d.operator.parametric:
            CO = np.ones(1)
        else:
            CO = np.array(d.operator.evaluate_coefficients(mu))

        est = [np.array([0.])]
        max_k = len(U) - 1 if k is None else k
        for i in xrange(max_k):
            C = np.hstack((np.dot(CM[..., np.newaxis], (U.data[i + 1] - U.data[i])[np.newaxis, ...]).ravel(), CR,
                           np.dot(CO[..., np.newaxis], U.data[i + 1][np.newaxis, ...]).ravel()))
            C = NumpyVectorArray(C)
            C = self.estimator_matrix.pairwise_apply2(C, C)

            if C > 0.:
                est.append(est[i] + C)
            elif 0. > C > -1e-10:
                est.append(est[i])
            else:
                ValueError('norm is negative (square = {})'.format(C))

        est = np.array(est)
        dt = d.T / discretization.time_stepper.nt
        est *= dt

        est /= 2. - self.gamma

        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)

        if not return_error_trajectory:
            return np.sqrt(est[-1])
        else:
            return NumpyVectorArray(np.sqrt(est))

    def restricted_to_subbasis(self, dim, discretization):
        d = discretization
        cr = 1 if not d.rhs.parametric else len(d.rhs.operators)
        co = 1 if not d.operator.parametric else len(d.operator.operators)
        old_dim = d.operator.source.dim

        indices = np.concatenate((np.arange(dim),
                                  np.arange(cr) + old_dim,
                                  ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel() + old_dim + cr))
        matrix = self.estimator_matrix._matrix[indices, :][:, indices]

        return ReduceParabolicSimpleEnergyEstimator(NumpyMatrixOperator(matrix), self.coercivity_estimator,
                                                              self.gamma)

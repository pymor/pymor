# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains algorithms for the empirical interpolation of operators.

The main work for generating the necessary interpolation data is handled by
the :func:`ei_greedy` method. The objects returned by this method can be used
to instantiate an |EmpiricalInterpolatedOperator|.

As a convenience, the :func:`interpolate_operators` method allows to perform
the empirical interpolation of the |Operators| of a given discretization with
a single function call.
"""

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.logger import getLogger
from pymor.algorithms.pod import pod
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interfaces import RemoteObjectInterface
from pymor.parallel.manager import RemoteObjectManager
from pymor.vectorarrays.interfaces import VectorArrayInterface


def ei_greedy(U, error_norm=None, atol=None, rtol=None, max_interpolation_dofs=None,
              projection='ei', product=None, copy=True, pool=dummy_pool):
    """Generate data for empirical interpolation by a greedy search (EI-Greedy algorithm).

    Given a |VectorArray| `U`, this method generates a collateral basis and
    interpolation DOFs for empirical interpolation of the vectors contained in `U`.
    The returned objects can also be used to instantiate an |EmpiricalInterpolatedOperator|.

    The interpolation data is generated by a greedy search algorithm, adding in each
    loop the worst approximated vector in `U` to the collateral basis.

    Parameters
    ----------
    U
        A |VectorArray| of vectors to interpolate.
    error_norm
        Norm w.r.t. which to calculate the interpolation error. If `None`, the Euclidean norm
        is used.
    atol
        Stop the greedy search if the largest approximation error is below this threshold.
    rtol
        Stop the greedy search if the largest relative approximation error is below this threshold.
    max_interpolation_dofs
        Stop the greedy search if the number of interpolation DOF (= dimension of the collateral
        basis) reaches this value.
    projection
        If `ei`, compute the approximation error by comparing the given vector by its
        interpolant. If `orthogonal`, compute the error by comparing with the orthogonal projection
        onto the span of the collateral basis.
    product
        If `projection == 'orthogonal'`, the product which is used to perform the projection.
        If `None`, the Euclidean product is used.
    copy
        If `False`, `U` will be modified during executing of the algorithm.
    pool
        If not `None`, the |WorkerPool| to use for parallelization.

    Returns
    -------
    interpolation_dofs
        |NumPy array| of the DOFs at which the vectors are evaluated.
    collateral_basis
        |VectorArray| containing the generated collateral basis.
    data
        Dict containing the following fields:

            :errors:                Sequence of maximum approximation errors during
                                    greedy search.
            :triangularity_errors:  Sequence of maximum absolute values of interoplation
                                    matrix coefficients in the upper triangle (should
                                    be near zero).
    """

    assert projection in ('orthogonal', 'ei')

    if pool:  # dispatch to parallel implemenation
        if projection == 'ei':
            pass
        elif projection == 'orthogonal':
            raise ValueError('orthogonal projection not supported in parallel implementation')
        else:
            assert False
        assert isinstance(U, (VectorArrayInterface, RemoteObjectInterface))
        with RemoteObjectManager() as rom:
            if isinstance(U, VectorArrayInterface):
                U = rom.manage(pool.scatter_array(U))
            return _parallel_ei_greedy(U, error_norm=error_norm, atol=atol, rtol=rtol,
                                       max_interpolation_dofs=max_interpolation_dofs, copy=copy, pool=pool)

    assert isinstance(U, VectorArrayInterface)

    logger = getLogger('pymor.algorithms.ei.ei_greedy')
    logger.info('Generating Interpolation Data ...')

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    collateral_basis = U.empty()
    max_errs = []
    triangularity_errs = []

    if copy:
        U = U.copy()

    if projection == 'orthogonal':
        ERR = U.copy()
        onb_collateral_basis = collateral_basis.empty()
    else:
        ERR = U

    errs = ERR.l2_norm() if error_norm is None else error_norm(ERR)
    max_err_ind = np.argmax(errs)
    initial_max_err = max_err = errs[max_err_ind]

    # main loop
    while True:
        if max_interpolation_dofs is not None and len(interpolation_dofs) >= max_interpolation_dofs:
            logger.info('Maximum number of interpolation DOFs reached. Stopping extension loop.')
            logger.info('Final maximum {} error with {} interpolation DOFs: {}'.format(
                'projection' if projection else 'interpolation', len(interpolation_dofs), max_err))
            break

        logger.info('Maximum {} error with {} interpolation DOFs: {}'
                    .format('projection' if projection else 'interpolation',
                            len(interpolation_dofs), max_err))

        if atol is not None and max_err <= atol:
            logger.info('Absolute error tolerance reached! Stopping extension loop.')
            break

        if rtol is not None and max_err / initial_max_err <= rtol:
            logger.info('Relative error tolerance reached! Stopping extension loop.')
            break

        # compute new interpolation dof and collateral basis vector
        new_vec = U.copy(ind=max_err_ind)
        new_dof = new_vec.amax()[0][0]
        if new_dof in interpolation_dofs:
            logger.info('DOF {} selected twice for interplation! Stopping extension loop.'.format(new_dof))
            break
        new_dof_value = new_vec.components([new_dof])[0, 0]
        if new_dof_value == 0.:
            logger.info('DOF {} selected for interpolation has zero maximum error! Stopping extension loop.'
                        .format(new_dof))
            break
        new_vec *= 1 / new_dof_value
        interpolation_dofs = np.hstack((interpolation_dofs, new_dof))
        collateral_basis.append(new_vec)
        max_errs.append(max_err)

        # update U and ERR
        new_dof_values = U.components([new_dof])
        U.axpy(-new_dof_values[:, 0], new_vec)
        if projection == 'orthogonal':
            onb_collateral_basis.append(new_vec)
            gram_schmidt(onb_collateral_basis, offset=len(onb_collateral_basis) - 1, copy=False)
            coeffs = ERR.dot(onb_collateral_basis, o_ind=len(onb_collateral_basis) - 1)
            ERR.axpy(-coeffs[:, 0], onb_collateral_basis, x_ind=len(onb_collateral_basis) - 1)
        errs = ERR.l2_norm() if error_norm is None else error_norm(ERR)
        max_err_ind = np.argmax(errs)
        max_err = errs[max_err_ind]

    interpolation_matrix = collateral_basis.components(interpolation_dofs).T
    triangularity_errors = np.abs(interpolation_matrix - np.tril(interpolation_matrix))
    for d in range(1, len(interpolation_matrix) + 1):
        triangularity_errs.append(np.max(triangularity_errors[:d, :d]))

    if len(triangularity_errs) > 0:
        logger.info('Interpolation matrix is not lower triangular with maximum error of {}'
                    .format(triangularity_errs[-1]))
        logger.info('')

    data = {'errors': max_errs, 'triangularity_errors': triangularity_errs}

    return interpolation_dofs, collateral_basis, data


def deim(U, modes=None, error_norm=None, product=None):
    """Generate data for empirical interpolation using DEIM algorithm.

    Given a |VectorArray| `U`, this method generates a collateral basis and
    interpolation DOFs for empirical interpolation of the vectors contained in `U`.
    The returned objects can also be used to instantiate an |EmpiricalInterpolatedOperator|.

    The collateral basis is determined by the first :func:`~pymor.algorithms.pod.pod` modes of `U`.

    Parameters
    ----------
    U
        A |VectorArray| of vectors to interpolate.
    modes
        Dimension of the collateral basis i.e. number of POD modes of the vectors in `U`.
    error_norm
        Norm w.r.t. which to calculate the interpolation error. If `None`, the Euclidean norm
        is used.
    product
        Product |Operator| used for POD.

    Returns
    -------
    interpolation_dofs
        |NumPy array| of the DOFs at which the vectors are interpolated.
    collateral_basis
        |VectorArray| containing the generated collateral basis.
    data
        Dict containing the following fields:

            :errors: Sequence of maximum approximation errors during greedy search.
    """

    assert isinstance(U, VectorArrayInterface)

    logger = getLogger('pymor.algorithms.ei.deim')
    logger.info('Generating Interpolation Data ...')

    collateral_basis = pod(U, modes, product=product)[0]

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    interpolation_matrix = np.zeros((0, 0))
    errs = []

    for i in range(len(collateral_basis)):

        if len(interpolation_dofs) > 0:
            coefficients = np.linalg.solve(interpolation_matrix,
                                           collateral_basis.components(interpolation_dofs, ind=i).T).T
            U_interpolated = collateral_basis.lincomb(coefficients, ind=list(range(len(interpolation_dofs))))
            ERR = collateral_basis.copy(ind=i)
            ERR -= U_interpolated
        else:
            ERR = collateral_basis.copy(ind=i)

        err = ERR.l2_norm() if error_norm is None else error_norm(ERR)

        logger.info('Interpolation error for basis vector {}: {}'.format(i, err))

        # compute new interpolation dof and collateral basis vector
        new_dof = ERR.amax()[0][0]

        if new_dof in interpolation_dofs:
            logger.info('DOF {} selected twice for interplation! Stopping extension loop.'.format(new_dof))
            break

        interpolation_dofs = np.hstack((interpolation_dofs, new_dof))
        interpolation_matrix = collateral_basis.components(interpolation_dofs, ind=list(range(len(interpolation_dofs)))).T
        errs.append(err)

        logger.info('')

    if len(interpolation_dofs) < len(collateral_basis):
        collateral_basis.remove(ind=list(range(len(interpolation_dofs), len(collateral_basis))))

    logger.info('Finished.'.format(new_dof))

    data = {'errors': errs}

    return interpolation_dofs, collateral_basis, data


def interpolate_operators(discretization, operator_names, parameter_sample, error_norm=None,
                          atol=None, rtol=None, max_interpolation_dofs=None,
                          projection='ei', product=None, pool=dummy_pool):
    """Empirical operator interpolation using the EI-Greedy algorithm.

    This is a convenience method for facilitating the use of :func:`ei_greedy`. Given
    a |Discretization|, names of |Operators|, and a sample of |Parameters|, first the operators
    are evaluated on the solution snapshots of the discretization for the provided parameters.
    These evaluations are then used as input for :func:`ei_greedy`. Finally the resulting
    interpolation data is used to create |EmpiricalInterpolatedOperators| and a new
    discretization with the interpolated operators is returned.

    Note that this implementation creates ONE common collateral basis for all specified
    operators which might not be what you want.

    Parameters
    ----------
    discretization
        The |Discretization| whose |Operators| will be interpolated.
    operator_names
        List of keys in the `operators` dict of the discretization. The corresponding
        |Operators| will be interpolated.
    parameter_sample
        A list of |Parameters| for which solution snapshots are calculated.
    error_norm
        See :func:`ei_greedy`.
    atol
        See :func:`ei_greedy`.
    rtol
        See :func:`ei_greedy`.
    max_interpolation_dofs
        See :func:`ei_greedy`.
    projection
        See :func:`ei_greedy`.
    product
        See :func:`ei_greedy`.
    pool
        If not `None`, the |WorkerPool| to use for parallelization.

    Returns
    -------
    ei_discretization
        |Discretization| with |Operators| given by `operator_names` replaced by
        |EmpiricalInterpolatedOperators|.
    data
        Dict containing the following fields:

            :dofs:    |NumPy array| of the DOFs at which the |Operators| have to be evaluated.
            :basis:   |VectorArray| containing the generated collateral basis.
            :errors:  Sequence of maximum approximation errors during greedy search.
    """

    logger = getLogger('pymor.algorithms.ei.interpolate_operators')
    logger.info('Computing operator evaluations on solution snapshots ...')
    operators = [discretization.operators[operator_name] for operator_name in operator_names]

    with RemoteObjectManager() as rom:
        if pool:
            logger.info('Using pool of {} workers for parallel evaluation'.format(len(pool)))
            evaluations = rom.manage(pool.push(discretization.solution_space.empty()))
            pool.map(_interpolate_operators_build_evaluations, parameter_sample,
                     d=discretization, operators=operators, evaluations=evaluations)
        else:
            evaluations = operators[0].range.empty()
            for mu in parameter_sample:
                U = discretization.solve(mu)
                for op in operators:
                    evaluations.append(op.apply(U, mu=mu))

        dofs, basis, data = ei_greedy(evaluations, error_norm, atol=atol, rtol=rtol,
                                      max_interpolation_dofs=max_interpolation_dofs,
                                      projection=projection, product=product, copy=False, pool=pool)

    ei_operators = {name: EmpiricalInterpolatedOperator(operator, dofs, basis, triangular=True)
                    for name, operator in zip(operator_names, operators)}
    operators_dict = discretization.operators.copy()
    operators_dict.update(ei_operators)
    ei_discretization = discretization.with_(operators=operators_dict, name='{}_ei'.format(discretization.name))

    data.update({'dofs': dofs, 'basis': basis})
    return ei_discretization, data


def _interpolate_operators_build_evaluations(mu, d=None, operators=None, evaluations=None):
    U = d.solve(mu)
    for op in operators:
        evaluations.append(op.apply(U, mu=mu))


def _parallel_ei_greedy(U, pool, error_norm=None, atol=None, rtol=None, max_interpolation_dofs=None, copy=True):

    assert isinstance(U, RemoteObjectInterface)

    logger = getLogger('pymor.algorithms.ei.ei_greedy')
    logger.info('Generating Interpolation Data ...')
    logger.info('Using pool of {} workers for parallel greedy search'.format(len(pool)))

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    collateral_basis = pool.apply_only(_parallel_ei_greedy_get_empty, 0, U=U)
    max_errs = []
    triangularity_errs = []

    with pool.push({}) as distributed_data:
        errs = pool.apply(_parallel_ei_greedy_initialize,
                          U=U, error_norm=error_norm, copy=copy, data=distributed_data)
        max_err_ind = np.argmax(errs)
        initial_max_err = max_err = errs[max_err_ind]

        # main loop
        while True:

            if max_interpolation_dofs is not None and len(interpolation_dofs) >= max_interpolation_dofs:
                logger.info('Maximum number of interpolation DOFs reached. Stopping extension loop.')
                logger.info('Final maximum interpolation error with {} interpolation DOFs: {}'
                            .format(len(interpolation_dofs), max_err))
                break

            logger.info('Maximum interpolation error with {} interpolation DOFs: {}'
                        .format(len(interpolation_dofs), max_err))

            if atol is not None and max_err <= atol:
                logger.info('Absolute error tolerance reached! Stopping extension loop.')
                break

            if rtol is not None and max_err / initial_max_err <= rtol:
                logger.info('Relative error tolerance reached! Stopping extension loop.')
                break

            # compute new interpolation dof and collateral basis vector
            new_vec = pool.apply_only(_parallel_ei_greedy_get_vector, max_err_ind, data=distributed_data)
            new_dof = new_vec.amax()[0][0]
            if new_dof in interpolation_dofs:
                logger.info('DOF {} selected twice for interplation! Stopping extension loop.'.format(new_dof))
                break
            new_dof_value = new_vec.components([new_dof])[0, 0]
            if new_dof_value == 0.:
                logger.info('DOF {} selected for interpolation has zero maximum error! Stopping extension loop.'
                            .format(new_dof))
                break
            new_vec *= 1 / new_dof_value
            interpolation_dofs = np.hstack((interpolation_dofs, new_dof))
            collateral_basis.append(new_vec)
            max_errs.append(max_err)

            errs = pool.apply(_parallel_ei_greedy_update, new_vec=new_vec, new_dof=new_dof, data=distributed_data)
            max_err_ind = np.argmax(errs)
            max_err = errs[max_err_ind]

    interpolation_matrix = collateral_basis.components(interpolation_dofs).T
    triangularity_errors = np.abs(interpolation_matrix - np.tril(interpolation_matrix))
    for d in range(1, len(interpolation_matrix) + 1):
        triangularity_errs.append(np.max(triangularity_errors[:d, :d]))

    if len(triangularity_errs) > 0:
        logger.info('Interpolation matrix is not lower triangular with maximum error of {}'
                    .format(triangularity_errs[-1]))
        logger.info('')

    data = {'errors': max_errs, 'triangularity_errors': triangularity_errs}

    return interpolation_dofs, collateral_basis, data


def _parallel_ei_greedy_get_empty(U=None):
    return U.empty()


def _parallel_ei_greedy_initialize(U=None, error_norm=None, copy=None, data=None):
    if copy:
        U = U.copy()
    data['U'] = U
    data['error_norm'] = error_norm
    errs = U.l2_norm() if error_norm is None else error_norm(U)
    data['max_err_ind'] = max_err_ind = np.argmax(errs)
    return errs[max_err_ind]


def _parallel_ei_greedy_get_vector(data=None):
    return data['U'].copy(ind=data['max_err_ind'])


def _parallel_ei_greedy_update(new_vec=None, new_dof=None, data=None):
    U = data['U']
    error_norm = data['error_norm']

    new_dof_values = U.components([new_dof])
    U.axpy(-new_dof_values[:, 0], new_vec)

    errs = U.l2_norm() if error_norm is None else error_norm(U)
    data['max_err_ind'] = max_err_ind = np.argmax(errs)
    return errs[max_err_ind]

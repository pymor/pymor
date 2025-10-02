# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import warnings
from numbers import Number

import numpy as np

from pymor.algorithms.line_search import armijo
from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError, NewtonError
from pymor.solvers.interface import Solver


class NewtonSolver(Solver):
    """Newton algorithm.

    Parameters
    ----------
    range_product
        The inner product with which the norm of the residual is computed.
        If `None`, the Euclidean inner product is used.
    source_product
        The inner product with which the norm of the solution and update
        vectors is computed. If `None`, the Euclidean inner product is used.
    miniter
        Minimum amount of iterations to perform.
    maxiter
        Fail if the iteration count reaches this value without converging.
    atol
        Finish when the error measure is below this threshold.
    rtol
        Finish when the error measure has been reduced by this factor
        relative to the norm of the initial residual resp. the norm of the current solution.
    relax
        If real valued, relaxation factor for Newton updates; otherwise `'armijo'` to
        indicate that the :func:`~pymor.algorithms.line_search.armijo` line search algorithm
        shall be used.
    line_search_params
        Dictionary of additional parameters passed to the line search method.
    stagnation_window
        Finish when the error measure has not been reduced by a factor of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.
    error_measure
        If `'residual'`, convergence depends on the norm of the residual. If
        `'update'`, convergence depends on the norm of the update vector.
    jacobian_solver
        The |Solver| to use for the linear update equations.
    return_stages
        If `True`, return a |VectorArray| of the intermediate approximations of `U`
        after each iteration.
    return_residuals
        If `True`, return a |VectorArray| of all residual vectors which have been computed
        during the Newton iterations.
    """

    least_squares = False

    @defaults('miniter', 'maxiter', 'rtol', 'atol', 'relax', 'stagnation_window', 'stagnation_threshold',
              'return_stages', 'return_residuals')
    def __init__(self, range_product=None, source_product=None,
                 miniter=0, maxiter=100, atol=0., rtol=1e-7, relax='armijo', line_search_params=None,
                 stagnation_window=3, stagnation_threshold=np.inf, error_measure='update',
                 jacobian_solver=None, return_stages=False, return_residuals=False):
        assert error_measure in ('residual', 'update')
        self.__auto_init(locals())

    def _solve(self, operator, V, mu, initial_guess):
        info = {'solution_norms': [], 'update_norms': [], 'residual_norms': []}
        if self.return_stages:
            info['stages'] = operator.source.empty()
        if self.return_residuals:
            info['residuals'] = operator.range.empty()
        U = V.empty(reserve=len(V))
        for i in range(len(V)):
            U.append(self._solve_one_rhs(operator, V[i],
                                         initial_guess=initial_guess[i] if initial_guess is not None else None,
                                         mu=mu, info=info),
                     remove_from_other=True)
        return U, info


    def _solve_one_rhs(self, operator, V, mu, initial_guess, info):
        range_product, source_product = self.range_product, self.source_product
        error_measure, atol, rtol, stagnation_threshold, stagnation_window = \
             self.error_measure, self.atol, self.rtol, self.stagnation_threshold, self.stagnation_window
        miniter, maxiter, relax = self.miniter, self.maxiter, self.relax

        if initial_guess is None:
            initial_guess = operator.source.zeros()

        U = initial_guess.copy()
        residual = V - operator.apply(U, mu=mu)

        # compute norms
        solution_norm = U.norm(source_product)[0]
        solution_norms = [solution_norm]
        update_norms = []
        residual_norm = residual.norm(range_product)[0]
        residual_norms = [residual_norm]

        # select error measure for convergence criteria
        err = residual_norm if error_measure == 'residual' else np.inf
        err_scale_factor = err
        errs = residual_norms if error_measure == 'residual' else update_norms

        self.logger.info(f'     norm:{solution_norm:.3e}                                 res:{residual_norm:.3e}')

        iteration = 0
        while True:
            # check for convergence / failure of convergence
            if iteration >= miniter:
                if residual_norm == 0:
                    # handle the corner case where error_norm == update, U is the exact solution
                    # and the jacobian of operator is not invertible at the exact solution
                    self.logger.info('Norm of residual exactly zero. Converged.')
                    break
                if err < atol:
                    self.logger.info(f'Absolute tolerance of {atol} for norm of {error_measure} reached. Converged.')
                    break
                if err < rtol * err_scale_factor:
                    self.logger.info(f'Relative tolerance of {rtol} for norm of {error_measure} reached. Converged.')
                    break
                if (len(errs) >= stagnation_window + 1
                        and err > stagnation_threshold * max(errs[-stagnation_window - 1:])):
                    self.logger.info(f'Norm of {error_measure} is stagnating (threshold: {stagnation_threshold:5e}, '
                                     f'window: {stagnation_window}). Converged.')
                    break
                if iteration >= maxiter:
                    raise NewtonError(f'Failed to converge after {iteration} iterations.')

            iteration += 1

            # store convergence history
            if iteration > 0 and 'stages' in info:
                info['stages'].append(U)
            if 'residuals' in info:
                info['residuals'].append(residual)

            # compute update
            jacobian = operator.jacobian(U, mu=mu)
            try:
                update = jacobian.apply_inverse(residual, solver=self.jacobian_solver)
            except InversionError as e:
                raise NewtonError('Could not invert jacobian.') from e

            # compute step size
            if isinstance(relax, Number):
                step_size = relax
            elif relax == 'armijo':
                def res(x):
                    residual_vec = V - operator.apply(x, mu=mu)
                    return residual_vec.norm(range_product)[0]

                if range_product is None:
                    grad = - (jacobian.apply(residual) + jacobian.apply_adjoint(residual))
                else:
                    grad = - (jacobian.apply_adjoint(range_product.apply(residual))
                              + jacobian.apply(range_product.apply_adjoint(residual)))
                step_size, _ = armijo(res, U, update, grad=grad, initial_value=residual_norm,
                                      **(self.line_search_params or {}))
            else:
                raise ValueError('Unknown line search method.')

            # update solution and residual
            U.axpy(step_size, update)
            residual = V - operator.apply(U, mu=mu)

            # compute norms
            solution_norm = U.norm(source_product)[0]
            solution_norms.append(solution_norm)
            update_norm = update.norm(source_product)[0] * step_size
            update_norms.append(update_norm)
            residual_norm = residual.norm(range_product)[0]
            residual_norms.append(residual_norm)

            # select error measure for next iteration
            err = residual_norm if error_measure == 'residual' else update_norm
            if error_measure == 'update':
                err_scale_factor = solution_norm

            with warnings.catch_warnings():
                # ignore division-by-zero warnings when solution_norm is zero
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                self.logger.info(f'it:{iteration} '
                                 f'norm:{solution_norm:.3e} '
                                 f'upd:{update_norm:.3e} '
                                 f'rel_upd:{update_norm / solution_norm:.3e} '
                                 f'res:{residual_norm:.3e} '
                                 f'red:{residual_norm / residual_norms[-2]:.3e} '
                                 f'tot_red:{residual_norm / residual_norms[0]:.3e}')

            if not np.isfinite(residual_norm) or not np.isfinite(solution_norm):
                raise NewtonError('Failed to converge.')

        self.logger.info('')

        info['solution_norms'].append(np.array(solution_norms))
        info['update_norms'].append(np.array(update_norms))
        info['residual_norms'].append(np.array(residual_norms))

        return U

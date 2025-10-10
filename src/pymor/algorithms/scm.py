# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Module for the successive constraints method."""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import KDTree

from pymor.algorithms.eigs import eigs
from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ConstantParameterFunctional, ParameterFunctional


class LBSuccessiveConstraintsFunctional(ParameterFunctional):
    """|ParameterFunctional| providing the lower bound from the successive constraints method.

    See :cite:`HRSP07`.

    Parameters
    ----------
    operator
        |LincombOperator| for which to provide a lower bound on the coercivity constant.
    constraint_parameters
        List of |Parameters| used to construct the constraints.
    coercivity_constants
        A list of coercivity constants for the `constraint_parameters`.
    bounds
        List of tuples containing lower and upper bounds for the design variables,
        i.e. the unknowns in the linear program.
    linprog_method
        Name of the algorithm to use for solving the linear program using `scipy.optimize.linprog`.
    linprog_options
        Dictionary of additional solver options passed to `scipy.optimize.linprog`.
    M
        Number of parameters to use for estimating the coercivity constant.
        The `M` closest parameters (with respect to the Euclidean distance) are chosen.
        If `None`, all parameters from `constraint_parameters` are used.
    """

    @defaults('linprog_method')
    def __init__(self, operator, constraint_parameters, coercivity_constants, bounds,
                 linprog_method='highs', linprog_options={}, M=None):
        assert isinstance(operator, LincombOperator)
        assert all(op.linear and not op.parametric for op in operator.operators)
        self.__auto_init(locals())
        self.operators = operator.operators
        self.thetas = tuple(ConstantParameterFunctional(f) if not isinstance(f, ParameterFunctional) else f
                            for f in operator.coefficients)

        if self.M is not None:
            if len(self.constraint_parameters) < self.M:
                self.logger.warning(f'Only {len(self.constraint_parameters)} parameters available, M is clipped ...')
                self.M = len(self.constraint_parameters)
            self.logger.info(f'Setting up KDTree to find {self.M} neighboring parameters ...')
            self.kdtree = KDTree(np.array([mu.to_numpy() for mu in self.constraint_parameters]))

        assert len(self.bounds) == len(self.operators)
        assert all(isinstance(b, tuple) and len(b) == 2 for b in self.bounds)

        assert len(self.coercivity_constants) == len(self.constraint_parameters)

    def evaluate(self, mu=None):
        c, A_ub, b_ub = self._construct_linear_program(mu)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=self.bounds,
                      method=self.linprog_method, options=self.linprog_options)
        return res['fun']

    def _construct_linear_program(self, mu):
        if self.M is not None:
            _, indices = self.kdtree.query(mu.to_numpy(), k=self.M)
            if isinstance(indices, np.int64):
                indices = [indices]
            selected_parameters = [self.constraint_parameters[i] for i in list(indices)]
        else:
            indices = np.arange(len(self.constraint_parameters))
            selected_parameters = self.constraint_parameters
        c = np.array([theta(mu) for theta in self.thetas])
        A_ub = - np.array([[theta(mu_con) for theta in self.thetas]
                           for mu_con in selected_parameters])
        b_ub = - np.array(self.coercivity_constants)[list(indices)]
        return c, A_ub, b_ub


class UBSuccessiveConstraintsFunctional(ParameterFunctional):
    """|ParameterFunctional| providing the upper bound from the successive constraints method.

    See :cite:`HRSP07`.

    Parameters
    ----------
    operator
        |LincombOperator| for which to provide an upper bound on the
        coercivity constant.
    constraint_parameters
        List of |Parameters| used to construct the constraints.
    minimizers
        List of minimizers associated to the coercivity constants of the
        operators in `operator`.
    """

    def __init__(self, operator, constraint_parameters, minimizers):
        assert isinstance(operator, LincombOperator)
        assert all(op.linear and not op.parametric for op in operator.operators)
        self.__auto_init(locals())
        self.operators = operator.operators
        self.thetas = tuple(ConstantParameterFunctional(f) if not isinstance(f, ParameterFunctional) else f
                            for f in operator.coefficients)

    def evaluate(self, mu=None):
        objective_values = [np.sum([theta(mu) * min_y for theta, min_y in zip(self.thetas, mins, strict=True)])
                            for mins in self.minimizers]
        return np.min(objective_values)


class SuccessiveConstraintsSurrogate(WeakGreedySurrogate):
    """Surrogate for constructing the functionals in a greedy algorithm.

    This surrogate is used in a weak greedy algorithm to select the parameters
    used to compute the constraints for the lower and upper bounds derived by
    the successive constraints method.

    Parameters
    ----------
    operator
        |LincombOperator| for which to provide a bounds on the
        coercivity constant.
    initial_parameter
        |Parameter| used to initialize the surrogate for the greedy algorithm.
    bounds
        List of tuples containing lower and upper bounds for the design variables,
        i.e. the unknowns in the linear program.
    product
        Product with respect to which the coercivity constant should be
        estimated.
    linprog_method
        Name of the algorithm to use for solving the linear program using `scipy.optimize.linprog`.
    linprog_options
        Dictionary of additional solver options passed to `scipy.optimize.linprog`.
    M
        Number of parameters to use for estimating the coercivity constant.
        The `M` closest parameters (with respect to the Euclidean distance) are chosen.
        If `None`, all parameters are used.
    """

    def __init__(self, operator, initial_parameter, bounds, product=None,
                 linprog_method='highs', linprog_options={}, M=None):
        self.__auto_init(locals())
        self.constraint_parameters = []
        self.coercivity_constants = []
        self.minimizers = []
        self.extend(initial_parameter)

    def evaluate(self, mus, return_all_values=False):
        evals_ub = np.array([self.ub_functional.evaluate(mu) for mu in mus])
        evals_lb = np.array([self.lb_functional.evaluate(mu) for mu in mus])
        estimated_errors = (evals_ub - evals_lb) / evals_ub
        if return_all_values:
            return estimated_errors
        else:
            index_max_error = np.argmax(estimated_errors)
            return estimated_errors[index_max_error], mus[index_max_error]

    def extend(self, mu):
        self.constraint_parameters.append(mu)
        fixed_parameter_op = self.operator.assemble(mu)
        eigvals, eigvecs = eigs(fixed_parameter_op, k=1, sigma=0, which='LM', E=self.product)
        self.coercivity_constants.append(eigvals[0].real)
        minimizer = eigvecs[0]
        minimizer_squared_norm = minimizer.norm(product=self.product) ** 2
        y_opt = np.array([op.apply2(minimizer, minimizer)[0].real / minimizer_squared_norm
                          for op in self.operator.operators])
        self.minimizers.append(y_opt)
        self.lb_functional = LBSuccessiveConstraintsFunctional(self.operator, self.constraint_parameters,
                                                               self.coercivity_constants, self.bounds,
                                                               linprog_method=self.linprog_method,
                                                               linprog_options=self.linprog_options, M=self.M)
        self.ub_functional = UBSuccessiveConstraintsFunctional(self.operator, self.constraint_parameters,
                                                               self.minimizers)


def construct_scm_functionals(operator, training_set, initial_parameter, atol=None, rtol=None, max_extensions=None,
                              product=None, linprog_method='highs', linprog_options={}, M=None):
    """Method to construct lower and upper bounds using the successive constraints method.

    Parameters
    ----------
    operator
        |LincombOperator| for which to provide a bounds on the
        coercivity constant.
    training_set
        |Parameters| used as training set for the greedy algorithm.
    initial_parameter
        |Parameter| used to initialize the surrogate for the greedy algorithm.
    atol
        If not `None`, stop the greedy algorithm if the maximum (estimated)
        error on the training set drops below this value.
    rtol
        If not `None`, stop the greedy algorithm if the maximum (estimated)
        relative error on the training set drops below this value.
    max_extensions
        If not `None`, stop the greedy algorithm after `max_extensions`
        extension steps.
    product
        Product with respect to which the coercivity constant should be
        estimated.
    linprog_method
        Name of the algorithm to use for solving the linear program using `scipy.optimize.linprog`.
    linprog_options
        Dictionary of additional solver options passed to `scipy.optimize.linprog`.
    M
        Number of parameters to use for estimating the coercivity constant.
        The `M` closest parameters (with respect to the Euclidean distance) are chosen.
        If `None`, all parameters selected in the greedy method are used.

    Returns
    -------
    Functional for a lower bound on the coercivity constant, functional
    for an upper bound on the coercivity constant, and the results returned
    by the weak greedy algorithm.
    """
    assert isinstance(operator, LincombOperator)
    assert all(op.linear and not op.parametric for op in operator.operators)

    logger = getLogger('pymor.algorithms.construct_scm_functionals')

    with logger.block('Computing bounds on design variables by solving eigenvalue problems ...'):
        def lower_upper_bound(operator):
            # some dispatch should be added here in the future
            eigvals, _ = eigs(operator, k=1, which='LM', E=product)
            largest = abs(eigvals[0].real)

            # use -|largest mag ev| as lower bound for shift-invert mode
            eigvals, _ = eigs(operator, k=1, sigma=-largest, which='LM', E=product)
            smallest = eigvals[0].real

            return smallest, largest

        bounds = [lower_upper_bound(aq) for aq in operator.operators]

    with logger.block('Running greedy algorithm to construct functionals ...'):
        surrogate = SuccessiveConstraintsSurrogate(operator, initial_parameter, bounds, product=product,
                                                   linprog_method=linprog_method, linprog_options=linprog_options, M=M)

        greedy_results = weak_greedy(surrogate, training_set, atol=atol, rtol=rtol, max_extensions=max_extensions)

    return surrogate.lb_functional, surrogate.ub_functional, greedy_results

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Module for the successive constraints method."""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import KDTree

from pymor.algorithms.eigs import eigs
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
        Either `None` or a list of tuples containing lower and upper bounds
        for the design variables, i.e. the unknowns in the linear program.
    linprog_method
        Name of the algorithm to use for solving the linear program using `scipy.optimize.linprog`.
    linprog_options
        Dictionary of additional solver options passed to `scipy.optimize.linprog`.
    M
        Number of parameters from `constraint_parameters` to use for estimating the coercivity
        constant. The `M` closest parameters (with respect to the Euclidean distance) are chosen.
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
        |LincombOperator| for which to provide an upper bound on the coercivity constant.
    constraint_parameters
        List of |Parameters| used to construct the constraints.
    minimizers
    """

    def __init__(self, operator, constraint_parameters, minimizers):
        assert isinstance(operator, LincombOperator)
        assert all(op.linear and not op.parametric for op in operator.operators)
        self.__auto_init(locals())
        self.operators = operator.operators
        self.thetas = tuple(ConstantParameterFunctional(f) if not isinstance(f, ParameterFunctional) else f
                            for f in operator.coefficients)

    def evaluate(self, mu=None):
        objective_values = [np.sum([theta(mu) * min_y for theta, min_y in zip(self.thetas, mins)])
                            for mins in self.minimizers]
        return np.min(objective_values)


def construct_scm_functionals(operator, constraint_parameters, linprog_method='highs', linprog_options={}, M=None):
    assert isinstance(operator, LincombOperator)
    assert all(op.linear and not op.parametric for op in operator.operators)
    operators = operator.operators

    logger = getLogger('pymor.algorithms.construct_scm_functionals')

    minimizers = []
    coercivity_constants = []

    with logger.block('Computing coercivity constants for parameters by solving eigenvalue problems ...'):
        for mu in constraint_parameters:
            fixed_parameter_op = operator.assemble(mu)
            eigvals, eigvecs = eigs(fixed_parameter_op, k=1, which='SM')
            coercivity_constants.append(eigvals[0].real)
            minimizer = eigvecs[0]
            minimizer_squared_norm = minimizer.norm() ** 2
            y_opt = np.array([op.apply2(minimizer, minimizer)[0].real / minimizer_squared_norm
                              for op in operators])
            minimizers.append(y_opt)

    with logger.block('Computing bounds on design variables by solving eigenvalue problems ...'):
        def lower_bound(operator):
            # some dispatch should be added here in the future
            eigvals, _ = eigs(operator, k=1, which='SM')
            return eigvals[0].real

        def upper_bound(operator):
            eigvals, _ = eigs(operator, k=1, which='LM')
            return eigvals[0].real

        bounds = [(lower_bound(aq), upper_bound(aq)) for aq in operators]

    lb_functional = LBSuccessiveConstraintsFunctional(operator, constraint_parameters, coercivity_constants, bounds,
                                                      linprog_method=linprog_method, linprog_options=linprog_options,
                                                      M=M)
    ub_functional = UBSuccessiveConstraintsFunctional(operator, constraint_parameters, minimizers)
    return lb_functional, ub_functional

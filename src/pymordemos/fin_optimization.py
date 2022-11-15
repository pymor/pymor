#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
from scipy.sparse import diags
from typer import run, Argument
from functools import partial
from scipy.optimize import minimize
from time import perf_counter
from pkg_resources import resource_filename

from pymor.basic import *
from pymor.parameters.functionals import (ConstantParameterFunctional, LincombParameterFunctional,
                                          MinThetaParameterFunctional)
from pymor.operators.constructions import BilinearFunctional
from pymor.discretizers.builtin.cg import RobinBoundaryOperator
from pymor.tools.random import new_rng
from pymor.algorithms.greedy import rb_greedy

from pymordemos.linear_optimization import report, record_results


def main(
    training_samples: int = Argument(3, help='Number of samples used for training the reduced basis.')
):
    """Example script for solving thermal fin PDE-constraind parameter optimization problems"""
    fom, parameter_space, mu_bar, range_1, range_2 = create_fom()

    initial_guess = np.array([0.25, 0.5])

    def fom_objective_functional(mu):
        return fom.output(mu)[0, 0]

    def fom_gradient_of_functional(mu):
        return fom.output_d_mu(fom.parameters.parse(mu), return_array=True, use_adjoint=True)

    opt_fom_minimization_data = {'num_evals': 0,
                                 'evaluations': [],
                                 'evaluation_points': [],
                                 'time': np.inf}

    tic = perf_counter()
    opt_fom_result = minimize(partial(record_results, fom_objective_functional,
                                      fom.parameters.parse, opt_fom_minimization_data),
                              initial_guess,
                              method='L-BFGS-B',
                              jac=fom_gradient_of_functional,
                              bounds=(range_1, range_2),
                              options={'ftol': 1e-15})
    opt_fom_minimization_data['time'] = perf_counter()-tic

    reference_mu = opt_fom_result.x

    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)

    training_set = parameter_space.sample_uniformly(training_samples)

    RB_reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    RB_greedy_data = rb_greedy(fom, RB_reductor, training_set, atol=1e-2)
    rom = RB_greedy_data['rom']

    def rom_objective_functional(mu):
        return rom.output(mu)[0, 0]

    def rom_gradient_of_functional(mu):
        return rom.output_d_mu(fom.parameters.parse(mu), return_array=True, use_adjoint=True)

    opt_rom_minimization_data = {'num_evals': 0,
                                 'evaluations': [],
                                 'evaluation_points': [],
                                 'time': np.inf,
                                 'offline_time': RB_greedy_data['time']}

    tic = perf_counter()
    opt_rom_result = minimize(partial(record_results, rom_objective_functional,
                                      fom.parameters.parse, opt_rom_minimization_data),
                              initial_guess,
                              method='L-BFGS-B',
                              jac=rom_gradient_of_functional,
                              bounds=(range_1, range_2),
                              options={'ftol': 1e-15})
    opt_rom_minimization_data['time'] = perf_counter()-tic

    print("\nResult of optimization with FOM model and adjoint gradient")
    report(opt_fom_result, fom.parameters.parse, opt_fom_minimization_data, reference_mu)
    print("Result of optimization with ROM model and adjoint gradient")
    report(opt_rom_result, fom.parameters.parse, opt_rom_minimization_data, reference_mu)


def create_fom():
    rng = new_rng(222)
    functions = [ExpressionFunction('(2.5 <= x[0]) * (x[0] <= 3.5) * (0 <= x[1]) * (x[1] <=4)* 1.', dim_domain=2),
                 ExpressionFunction('(0 <= x[0]) * (x[0] < 2.5) * (0.75 <= x[1]) * (x[1] <= 1) *1. \
                                    + (3.5 < x[0]) * (x[0] <= 6) * (0.75 <= x[1]) * (x[1] <= 1)* 1. \
                                    + (0 <= x[0]) * (x[0] < 2.5) * (1.75 <= x[1]) * (x[1] <= 2) * 1. \
                                    + (3.5 < x[0]) * (x[0] <= 6) * (1.75 <= x[1]) * (x[1] <= 2) * 1. \
                                    + (0 <= x[0]) * (x[0] < 2.5) * (2.75 <= x[1]) * (x[1] <= 3) *1. \
                                    + (3.5 < x[0]) * (x[0] <= 6) * (2.75 <= x[1]) * (x[1] <= 3) * 1. \
                                    + (0 <= x[0]) * (x[0] < 2.5) * (3.75 <= x[1]) * (x[1] <= 4) *1. \
                                    + (3.5 < x[0]) * (x[0] <= 6) * (3.75 <= x[1]) * (x[1] <= 4) * 1.', dim_domain=2)]
    coefficients = [1, ProjectionParameterFunctional('k', 1)]
    diffusion = LincombFunction(functions, coefficients)
    parameter_ranges = {
        'biot': (0.01, 1),
        'k': (0.1, 10)
    }
    domain = RectDomain([[0, 0], [6, 4]])
    helper_problem = StationaryProblem(
        domain=domain,
        diffusion=diffusion,
        rhs=ConstantFunction(0, 2),
        neumann_data=ConstantFunction(-1, 2),
        robin_data=(
            LincombFunction([ConstantFunction(1, 2)], [ProjectionParameterFunctional('biot', 1)]),
            ConstantFunction(0, 2)
        ), parameter_ranges=parameter_ranges
    )
    parameter_space = helper_problem.parameter_space
    mu_bar = {}
    mu_bar_helper = parameter_space.sample_uniformly(1)[0]
    for key in helper_problem.parameters:
        range_ = parameter_space.ranges[key]
        if range_[0] == 0:
            value = 10**(np.log10(range_[1]) / 2)
        else:
            value = 10**((np.log10(range_[0]) + np.log10(range_[1])) / 2)
        mu_bar[key] = [value for i in range(len(mu_bar_helper[key]))]
    mu_bar = helper_problem.parameters.parse(mu_bar)
    with rng:
        mu_d = parameter_space.sample_randomly(1)[0].to_numpy()
    mu_d[0] = np.array(0.01)
    mu_d[1] = np.array(0.1)
    mu_d = helper_problem.parameters.parse(mu_d)
    grid, boundary_info = load_gmsh(resource_filename('pymordemos', 'data/fin_mesh.msh'))
    helper_fom, data = discretize_stationary_cg(
        helper_problem, grid=grid, boundary_info=boundary_info, mu_energy_product=mu_bar
    )
    u_d = helper_fom.solve(mu_d)
    boundary_functional = helper_fom.rhs.operators[1]
    T_root_d = boundary_functional.apply_adjoint(u_d)
    T_root_d_squared = T_root_d.to_numpy()[0][0]**2
    weights = helper_problem.parameters.parse(1 / mu_d.to_numpy() ** 2)
    state_functional = ConstantParameterFunctional(1.)
    parameter_functionals = []
    parameter_functionals.append(.5 * state_functional * T_root_d_squared)
    parameter_functionals.append(state_functional)

    def construct_der_dicts(parameters):
        def construct_inner_dict():
            inner_dict = {}
            for key, item in parameters.items():
                inner_array = np.array([], dtype='<U60')
                for i in range(item):
                    inner_array = np.append(inner_array, ['0'])
                inner_array = np.array(inner_array, dtype='<U60')
                inner_dict[key] = inner_array
            return inner_dict
        der_expr = construct_inner_dict()
        second_der_expr = {}
        for key, item in parameters.items():
            outer_array = np.empty(item, dtype=dict)
            second_der_expr[key] = outer_array
            for i in range(item):
                second_der_expr[key][i] = construct_inner_dict()
        return der_expr, second_der_expr

    for key, item in helper_problem.parameters.items():
        for i in range(item):
            der_expr, second_der_expr = construct_der_dicts(helper_problem.parameters)
            der_expr[key][i] = '{}*{}**2*({}[{}]-'.format(weights[key][i], 1., key, i) + '{}'.format(mu_d[key][i]) + ')'
            second_der_expr[key][i][key][i] = '{}*{}**2'.format(weights[key][i], 1.)
            parameter_functionals.append(
                ExpressionParameterFunctional(
                    '{}*{}**2*0.5*({}[{}]'.format(weights[key][i], 1., key, i)
                    + '-{}'.format(mu_d[key][i])+')**2', second_derivative_expressions=second_der_expr,
                    derivative_expressions=der_expr, parameters=helper_problem.parameters))
    coeffs = []
    for f in parameter_functionals:
        coeffs.append(1.)
    const_coeff = LincombParameterFunctional(parameter_functionals, coeffs)
    lin_coeff = -1. * T_root_d.to_numpy()[0][0]
    const_op = ConstantOperator(NumpyVectorSpace(1).ones(), boundary_functional.range)
    bilin_op = NumpyMatrixOperator(
        diags(boundary_functional.H.matrix[0]), source_id=boundary_functional.range.id,
        range_id=boundary_functional.range.id
    )
    bilin_op = BilinearFunctional(bilin_op)
    output_functional = LincombOperator([const_op, boundary_functional.H, bilin_op], [const_coeff, lin_coeff, .5])
    l2_boundary_product = RobinBoundaryOperator(
        grid, data['boundary_info'], name='l2_boundary_product',
        robin_data=(ConstantFunction(1, 2), ConstantFunction(1, 2))
    )
    helper_fom = helper_fom.with_(
        products=dict(
            opt=helper_fom.energy_product, l2_boundary=l2_boundary_product,
            **helper_fom.products), output_functional=output_functional
    )
    pde_opt_fom = StationaryModel(
        helper_fom.operator, helper_fom.rhs, output_functional=output_functional,
        products=helper_fom.products, error_estimator=helper_fom.error_estimator,
        visualizer=helper_fom.visualizer
    )
    range_1 = helper_problem.parameter_space.ranges['biot']
    range_2 = helper_problem.parameter_space.ranges['k']
    return pde_opt_fom, parameter_space, mu_bar, range_1, range_2


if __name__ == '__main__':
    run(main)

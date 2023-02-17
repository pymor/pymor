# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.basic import Mu, NumpyVectorSpace
from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.parameters.functionals import ExpressionParameterFunctional, ProjectionParameterFunctional

pytestmark = pytest.mark.builtin


def test_ProjectionParameterFunctional():
    pf = ProjectionParameterFunctional('mu', 2, 0)
    mu = Mu({'mu': (10, 2)})

    derivative_to_first_index = pf.d_mu('mu', 0)
    derivative_to_second_index = pf.d_mu('mu', 1)

    second_derivative_first_first = pf.d_mu('mu', 0).d_mu('mu', 0)
    second_derivative_first_second = pf.d_mu('mu', 0).d_mu('mu', 1)
    second_derivative_second_first = pf.d_mu('mu', 1).d_mu('mu', 0)
    second_derivative_second_second = pf.d_mu('mu', 1).d_mu('mu', 1)

    der_mu_1 = derivative_to_first_index.evaluate(mu)
    der_mu_2 = derivative_to_second_index.evaluate(mu)

    hes_mu_1_mu_1 = second_derivative_first_first.evaluate(mu)
    hes_mu_1_mu_2 = second_derivative_first_second.evaluate(mu)
    hes_mu_2_mu_1 = second_derivative_second_first.evaluate(mu)
    hes_mu_2_mu_2 = second_derivative_second_second.evaluate(mu)

    assert pf.evaluate(mu) == 10
    assert der_mu_1 == 1
    assert der_mu_2 == 0
    assert hes_mu_1_mu_1 == 0
    assert hes_mu_1_mu_2 == 0
    assert hes_mu_2_mu_1 == 0
    assert hes_mu_2_mu_2 == 0


def test_ExpressionParameterFunctional():
    dict_of_d_mus = {'mu': ['200 * mu[0]', '2 * mu[0]'], 'nu': ['cos(nu[0])']}

    dict_of_second_derivative = {
        'mu': [{'mu': ['200', '2'], 'nu': ['0']}, {'mu': ['2', '0'], 'nu': ['0']}],
        'nu': [{'mu': ['0', '0'], 'nu': ['-sin(nu[0])']}]
    }

    epf = ExpressionParameterFunctional('100 * mu[0]**2 + 2 * mu[1] * mu[0] + sin(nu[0])',
                                        {'mu': 2, 'nu': 1},
                                        'functional_with_derivative_and_second_derivative',
                                        dict_of_d_mus, dict_of_second_derivative)

    mu = Mu({'mu': [10, 2], 'nu': [0]})

    derivative_to_first_mu_index = epf.d_mu('mu', 0)
    derivative_to_second_mu_index = epf.d_mu('mu', 1)
    derivative_to_nu_index = epf.d_mu('nu', 0)

    der_mu_1 = derivative_to_first_mu_index.evaluate(mu)
    der_mu_2 = derivative_to_second_mu_index.evaluate(mu)
    der_nu = derivative_to_nu_index.evaluate(mu)

    second_derivative_first_mu_first_mu = epf.d_mu('mu', 0).d_mu('mu', 0)
    second_derivative_first_mu_second_mu = epf.d_mu('mu', 0).d_mu('mu', 1)
    second_derivative_first_mu_nu = epf.d_mu('mu', 0).d_mu('nu')
    second_derivative_second_mu_first_mu = epf.d_mu('mu', 1).d_mu('mu', 0)
    second_derivative_second_mu_second_mu = epf.d_mu('mu', 1).d_mu('mu', 1)
    second_derivative_second_mu_nu = epf.d_mu('mu', 1).d_mu('nu')
    second_derivative_nu_first_mu = epf.d_mu('nu').d_mu('mu', 0)
    second_derivative_nu_second_mu = epf.d_mu('nu').d_mu('mu', 1)
    second_derivative_nu_nu = epf.d_mu('nu').d_mu('nu')

    hes_mu_1_mu_1 = second_derivative_first_mu_first_mu.evaluate(mu)
    hes_mu_1_mu_2 = second_derivative_first_mu_second_mu.evaluate(mu)
    hes_mu_1_nu = second_derivative_first_mu_nu.evaluate(mu)
    hes_mu_2_mu_1 = second_derivative_second_mu_first_mu.evaluate(mu)
    hes_mu_2_mu_2 = second_derivative_second_mu_second_mu.evaluate(mu)
    hes_mu_2_nu = second_derivative_second_mu_nu.evaluate(mu)
    hes_nu_mu_1 = second_derivative_nu_first_mu.evaluate(mu)
    hes_nu_mu_2 = second_derivative_nu_second_mu.evaluate(mu)
    hes_nu_nu = second_derivative_nu_nu.evaluate(mu)

    assert epf.evaluate(mu) == 100*10**2 + 2*2*10 + 0
    assert der_mu_1 == 200 * 10
    assert der_mu_2 == 2 * 10
    assert der_nu == 1
    assert hes_mu_1_mu_1 == 200
    assert hes_mu_1_mu_2 == 2
    assert hes_mu_1_nu == 0
    assert hes_mu_2_mu_1 == 2
    assert hes_mu_2_mu_2 == 0
    assert hes_mu_2_nu == 0
    assert hes_nu_mu_1 == 0
    assert hes_nu_mu_2 == 0
    assert hes_nu_nu == -0


def test_simple_ProductParameterFunctional():
    pf = ProjectionParameterFunctional('mu', 2, 0)
    mu = Mu({'mu': (10, 2)})
    productf = pf * 2 * 3

    derivative_to_first_index = productf.d_mu('mu', 0)
    derivative_to_second_index = productf.d_mu('mu', 1)

    second_derivative_first_first = productf.d_mu('mu', 0).d_mu('mu', 0)
    second_derivative_first_second = productf.d_mu('mu', 0).d_mu('mu', 1)
    second_derivative_second_first = productf.d_mu('mu', 1).d_mu('mu', 0)
    second_derivative_second_second = productf.d_mu('mu', 1).d_mu('mu', 1)

    der_mu_1 = derivative_to_first_index.evaluate(mu)
    der_mu_2 = derivative_to_second_index.evaluate(mu)

    hes_mu_1_mu_1 = second_derivative_first_first.evaluate(mu)
    hes_mu_1_mu_2 = second_derivative_first_second.evaluate(mu)
    hes_mu_2_mu_1 = second_derivative_second_first.evaluate(mu)
    hes_mu_2_mu_2 = second_derivative_second_second.evaluate(mu)

    assert der_mu_1 == 2 * 3
    assert der_mu_2 == 0
    assert hes_mu_1_mu_1 == 0
    assert hes_mu_1_mu_2 == 0
    assert hes_mu_2_mu_1 == 0
    assert hes_mu_2_mu_2 == 0

    dict_of_d_mus = {'mu': ['2*mu']}

    dict_of_second_derivative = {'mu': [{'mu': ['2']}]}

    epf = ExpressionParameterFunctional('mu**2',
                                        {'mu': 1},
                                        'functional_with_derivative_and_second_derivative',
                                        dict_of_d_mus, dict_of_second_derivative)

    productf = epf * 2
    mu = Mu({'mu': 3})

    derivative_to_first_index = productf.d_mu('mu')

    second_derivative_first_first = productf.d_mu('mu').d_mu('mu')

    der_mu = derivative_to_first_index.evaluate(mu)

    hes_mu_mu = second_derivative_first_first.evaluate(mu)

    assert productf.evaluate(mu) == 2 * 3 ** 2
    assert der_mu == 2 * 2 * 3
    assert hes_mu_mu == 2 * 2


def test_ProductParameterFunctional():
    # Projection ParameterFunctional
    pf = ProjectionParameterFunctional('mu', 2, 0)
    # Expression ParameterFunctional
    dict_of_d_mus = {'nu': ['2*nu']}
    dict_of_second_derivative = {'nu': [{'nu': ['2']}]}
    epf = ExpressionParameterFunctional('nu**2', {'nu': 1},
                                        'expression_functional',
                                        dict_of_d_mus, dict_of_second_derivative)
    mu = Mu({'mu': (10, 2), 'nu': 3})

    productf = pf * epf * 2. * pf

    derivative_to_first_index = productf.d_mu('mu', 0)
    derivative_to_second_index = productf.d_mu('mu', 1)
    derivative_to_third_index = productf.d_mu('nu', 0)

    second_derivative_first_first = productf.d_mu('mu', 0).d_mu('mu', 0)
    second_derivative_first_second = productf.d_mu('mu', 0).d_mu('mu', 1)
    second_derivative_first_third = productf.d_mu('mu', 0).d_mu('nu', 0)
    second_derivative_second_first = productf.d_mu('mu', 1).d_mu('mu', 0)
    second_derivative_second_second = productf.d_mu('mu', 1).d_mu('mu', 1)
    second_derivative_second_third = productf.d_mu('mu', 1).d_mu('nu', 0)
    second_derivative_third_first = productf.d_mu('nu', 0).d_mu('mu', 0)
    second_derivative_third_second = productf.d_mu('nu', 0).d_mu('mu', 1)
    second_derivative_third_third = productf.d_mu('nu', 0).d_mu('nu', 0)

    der_mu_1 = derivative_to_first_index.evaluate(mu)
    der_mu_2 = derivative_to_second_index.evaluate(mu)
    der_nu_3 = derivative_to_third_index.evaluate(mu)

    hes_mu_1_mu_1 = second_derivative_first_first.evaluate(mu)
    hes_mu_1_mu_2 = second_derivative_first_second.evaluate(mu)
    hes_mu_1_nu_3 = second_derivative_first_third.evaluate(mu)
    hes_mu_2_mu_1 = second_derivative_second_first.evaluate(mu)
    hes_mu_2_mu_2 = second_derivative_second_second.evaluate(mu)
    hes_mu_2_nu_3 = second_derivative_second_third.evaluate(mu)
    hes_nu_3_mu_1 = second_derivative_third_first.evaluate(mu)
    hes_nu_3_mu_2 = second_derivative_third_second.evaluate(mu)
    hes_nu_3_nu_3 = second_derivative_third_third.evaluate(mu)

    # note that productf(mu,nu) = 2 * pf(mu)**2 * epf(nu)
    # and thus:
    #      d_mu productf(mu,nu) = 2 * 2 * pf(mu) * (d_mu pf)(mu) * epf(nu)
    #      d_nu productf(mu,nu) = 2 * pf(mu)**2 * (d_nu epf)(nu)
    # and so forth
    assert der_mu_1 == 2 * 2 * 10 * 1 * 9
    assert der_mu_2 == 0
    assert der_nu_3 == 2 * 10**2 * 6
    assert hes_mu_1_mu_1 == 2 * 2 * 9
    assert hes_mu_1_mu_2 == 0
    assert hes_mu_1_nu_3 == 2 * 2 * 10 * 1 * 6
    assert hes_mu_2_mu_1 == 0
    assert hes_mu_2_mu_2 == 0
    assert hes_mu_2_nu_3 == 0
    assert hes_nu_3_mu_1 == 2 * 2 * 10 * 1 * 6
    assert hes_nu_3_mu_2 == 0
    assert hes_nu_3_nu_3 == 2 * 10**2 * 2


def test_d_mu_of_LincombOperator():
    dict_of_d_mus = {'mu': ['100', '2 * mu[0]'], 'nu': ['cos(nu[0])']}

    dict_of_second_derivative = {
        'mu': [{'mu': ['0', '2'], 'nu': ['0']}, {'mu': ['2', '0'], 'nu': ['0']}],
        'nu': [{'mu': ['0', '0'], 'nu': ['-sin(nu[0])']}]
    }

    pf = ProjectionParameterFunctional('mu', 2, 0)
    epf = ExpressionParameterFunctional('100 * mu[0] + 2 * mu[1] * mu[0] + sin(nu[0])',
                                        {'mu': 2, 'nu': 1},
                                        'functional_with_derivative',
                                        dict_of_d_mus, dict_of_second_derivative)

    mu = Mu({'mu': [10, 2], 'nu': [0]})

    space = NumpyVectorSpace(1)
    zero_op = ZeroOperator(space, space)
    operators = [zero_op, zero_op, zero_op]
    coefficients = [1., pf, epf]

    operator = LincombOperator(operators, coefficients)

    op_sensitivity_to_first_mu = operator.d_mu('mu', 0)
    op_sensitivity_to_second_mu = operator.d_mu('mu', 1)
    op_sensitivity_to_nu = operator.d_mu('nu', 0)

    eval_mu_1 = op_sensitivity_to_first_mu.evaluate_coefficients(mu)
    eval_mu_2 = op_sensitivity_to_second_mu.evaluate_coefficients(mu)
    eval_nu = op_sensitivity_to_nu.evaluate_coefficients(mu)

    second_derivative_first_mu_first_mu = operator.d_mu('mu', 0).d_mu('mu', 0)
    second_derivative_first_mu_second_mu = operator.d_mu('mu', 0).d_mu('mu', 1)
    second_derivative_first_mu_nu = operator.d_mu('mu', 0).d_mu('nu')
    second_derivative_second_mu_first_mu = operator.d_mu('mu', 1).d_mu('mu', 0)
    second_derivative_second_mu_second_mu = operator.d_mu('mu', 1).d_mu('mu', 1)
    second_derivative_second_mu_nu = operator.d_mu('mu', 1).d_mu('nu')
    second_derivative_nu_first_mu = operator.d_mu('nu').d_mu('mu', 0)
    second_derivative_nu_second_mu = operator.d_mu('nu').d_mu('mu', 1)
    second_derivative_nu_nu = operator.d_mu('nu').d_mu('nu')

    hes_mu_1_mu_1 = second_derivative_first_mu_first_mu.evaluate_coefficients(mu)
    hes_mu_1_mu_2 = second_derivative_first_mu_second_mu.evaluate_coefficients(mu)
    hes_mu_1_nu = second_derivative_first_mu_nu.evaluate_coefficients(mu)
    hes_mu_2_mu_1 = second_derivative_second_mu_first_mu.evaluate_coefficients(mu)
    hes_mu_2_mu_2 = second_derivative_second_mu_second_mu.evaluate_coefficients(mu)
    hes_mu_2_nu = second_derivative_second_mu_nu.evaluate_coefficients(mu)
    hes_nu_mu_1 = second_derivative_nu_first_mu.evaluate_coefficients(mu)
    hes_nu_mu_2 = second_derivative_nu_second_mu.evaluate_coefficients(mu)
    hes_nu_nu = second_derivative_nu_nu.evaluate_coefficients(mu)

    assert operator.evaluate_coefficients(mu) == [1., 10, 1040.]
    assert eval_mu_1 == [0., 1., 100.]
    assert eval_mu_2 == [0., 0., 2. * 10]
    assert eval_nu == [0., 0., 1.]

    assert hes_mu_1_mu_1 == [0., 0., 0.]
    assert hes_mu_1_mu_2 == [0., 0., 2]
    assert hes_mu_1_nu == [0., 0., 0]
    assert hes_mu_2_mu_1 == [0., 0., 2]
    assert hes_mu_2_mu_2 == [0., 0., 0]
    assert hes_mu_2_nu == [0., 0., 0]
    assert hes_nu_mu_1 == [0., 0., 0]
    assert hes_nu_mu_2 == [0., 0., 0]
    assert hes_nu_nu == [0., 0., -0]


def run_test_on_model(model, mu, parameter_name=None):
    gradient_with_adjoint_approach = model.output_d_mu(mu, return_array=True, use_adjoint=True)
    gradient_with_sensitivities = model.output_d_mu(mu, return_array=True, use_adjoint=False)
    assert np.allclose(gradient_with_adjoint_approach, gradient_with_sensitivities)
    if parameter_name is not None:
        u_d_mu = model.solve_d_mu(parameter_name, 1, mu=mu).to_numpy()
        u_d_mu_ = model.compute(solution_d_mu=True, mu=mu)['solution_d_mu'][parameter_name][1].to_numpy()
        assert np.allclose(u_d_mu, u_d_mu_)


def test_output_d_mu():
    from pymordemos.linear_optimization import create_fom

    grid_intervals = 10
    training_samples = 3

    # creating multiple foms to test
    fom, _ = create_fom(grid_intervals, vector_valued_output=True)
    easy_fom, _ = create_fom(grid_intervals, vector_valued_output=False)
    complex_op_fom = easy_fom.with_(operator=easy_fom.operator.with_(
        operators=[op * (1+2j) for op in easy_fom.operator.operators]))
    complex_out_fom = easy_fom.with_(output_functional=easy_fom.output_functional.with_(
        operators=[op * (1+2j) for op in easy_fom.output_functional.operators]))
    quad_fom, _ = create_fom(grid_intervals, output_type='quadratic')

    test_foms = [complex_op_fom, complex_out_fom, quad_fom]

    parameter_space = fom.parameters.space(0, np.pi)
    training_set = parameter_space.sample_uniformly(training_samples)

    # verifying that the adjoint and sensitivity gradients are the same and that solve_d_mu works
    for mu in training_set:
        run_test_on_model(fom, mu, 'diffusion')
        for model in test_foms:
            run_test_on_model(model, mu)

    # another fom to test the 3d case
    ops, coefs = fom.operator.operators, fom.operator.coefficients
    ops += (fom.operator.operators[1],)
    coefs += (ProjectionParameterFunctional('nu', 1, 0),)
    fom_ = fom.with_(operator=LincombOperator(ops, coefs))
    parameter_space = fom_.parameters.space(0, np.pi)
    training_set = parameter_space.sample_uniformly(training_samples)
    for mu in training_set:
        run_test_on_model(fom_, mu)

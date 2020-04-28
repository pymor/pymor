# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymortests.base import runmodule
import pytest

import numpy as np

from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.basic import NumpyVectorSpace, Parameter


def test_ProjectionParameterFunctional():
    pf = ProjectionParameterFunctional('mu', (2,), (0,))
    mu = Parameter({'mu': (10,2)})

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
    dict_of_d_mus = {'mu': ['200 * mu[0]', '2 * mu[0]'], 'nu': 'cos(nu)'}

    dict_of_second_derivative = {'mu': [{'mu': ['200', '2'], 'nu': '0'},
                              {'mu': ['2', '0'], 'nu': '0'}],
                       'nu': {'mu': ['0', '0'], 'nu': '-sin(nu)'}}

    epf = ExpressionParameterFunctional('100 * mu[0]**2 + 2 * mu[1] * mu[0] + sin(nu)',
                                        {'mu': (2,), 'nu': ()},
                                        'functional_with_derivative_and_second_derivative',
                                        dict_of_d_mus, dict_of_second_derivative)

    mu = Parameter({'mu': (10,2), 'nu': 0})

    derivative_to_first_mu_index = epf.d_mu('mu', 0)
    derivative_to_second_mu_index = epf.d_mu('mu', 1)
    derivative_to_nu_index = epf.d_mu('nu', ())

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

def test_ExpressionParameterFunctional_for_2d_array():
    dict_of_d_mus_2d = {'mu': [['10', '20'],['1', '2']], 'nu': 'cos(nu)'}

    epf2d = ExpressionParameterFunctional('10 * mu[(0,0)] + 20 * mu[(0,1)] \
                                          + 1 * mu[(1,0)] + 2 *  mu[(1,1)] \
                                          + sin(nu)',
                                        {'mu': (2,2), 'nu': ()},
                                        'functional_with_derivative',
                                        dict_of_d_mus_2d)

    derivative_to_11_mu_index = epf2d.d_mu('mu', (0,0))
    derivative_to_12_mu_index = epf2d.d_mu('mu', (0,1))
    derivative_to_21_mu_index = epf2d.d_mu('mu', (1,0))
    derivative_to_22_mu_index = epf2d.d_mu('mu', (1,1))
    derivative_to_nu_index = epf2d.d_mu('nu')

    mu = Parameter({'mu': [[1,2],[3,4]], 'nu': 0})

    der_mu_11 = derivative_to_11_mu_index.evaluate(mu)
    der_mu_12 = derivative_to_12_mu_index.evaluate(mu)
    der_mu_21 = derivative_to_21_mu_index.evaluate(mu)
    der_mu_22 = derivative_to_22_mu_index.evaluate(mu)
    der_nu = derivative_to_nu_index.evaluate(mu)

    assert epf2d.evaluate(mu) == 1*10 + 2*20 + 3*1 + 4*2 + 0
    assert der_mu_11 == 10
    assert der_mu_12 == 20
    assert der_mu_21 == 1
    assert der_mu_22 == 2
    assert der_nu == 1

def test_d_mu_of_LincombOperator():
    dict_of_d_mus = {'mu': ['100', '2 * mu[0]'], 'nu': 'cos(nu)'}

    dict_of_second_derivative = {'mu': [{'mu': ['0', '2'], 'nu': '0'},
                              {'mu': ['2', '0'], 'nu': '0'}],
                       'nu': {'mu': ['0', '0'], 'nu': '-sin(nu)'}}

    pf = ProjectionParameterFunctional('mu', (2,), (0,))
    epf = ExpressionParameterFunctional('100 * mu[0] + 2 * mu[1] * mu[0] + sin(nu)',
                                        {'mu': (2,), 'nu': ()},
                                        'functional_with_derivative',
                                        dict_of_d_mus, dict_of_second_derivative)

    mu = Parameter({'mu': (10,2), 'nu': 0})

    space = NumpyVectorSpace(1)
    zero_op = ZeroOperator(space, space)
    operators = [zero_op, zero_op, zero_op]
    coefficients = [1., pf, epf]

    operator = LincombOperator(operators, coefficients)

    op_sensitivity_to_first_mu = operator.d_mu('mu', 0)
    op_sensitivity_to_second_mu = operator.d_mu('mu', 1)
    op_sensitivity_to_nu = operator.d_mu('nu', ())

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

    assert hes_mu_1_mu_1 == [0. ,0. , 0.]
    assert hes_mu_1_mu_2 == [0. ,0. ,2]
    assert hes_mu_1_nu == [0. ,0. ,0]
    assert hes_mu_2_mu_1 == [0. ,0. ,2]
    assert hes_mu_2_mu_2 == [0. ,0. ,0]
    assert hes_mu_2_nu == [0. ,0. ,0]
    assert hes_nu_mu_1 == [0. ,0. ,0]
    assert hes_nu_mu_2 == [0. ,0. ,0]
    assert hes_nu_nu == [0. ,0. ,-0]

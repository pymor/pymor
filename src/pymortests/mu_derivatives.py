# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional

'''
test for ProjectionParameterFunctional
'''

pf = ProjectionParameterFunctional('mu', (2,), (0,))

mu = {'mu': (10,2)}

derivative_to_first_coordinate = pf.partial_derivative('mu', 0)
derivative_to_second_coordinate = pf.partial_derivative('mu', 1)

der_mu_1 = derivative_to_first_coordinate.evaluate(mu)
der_mu_2 = derivative_to_second_coordinate.evaluate(mu)

assert pf.evaluate(mu) == 10
assert der_mu_1 == 1
assert der_mu_2 == 0


'''
test for ExpressionParameterFunctional
'''

dict_of_partial_derivatives = {'mu': ['100', '2'], 'nu': 'cos(nu)'}

epf = ExpressionParameterFunctional('100 * mu[0] + 2 * mu[1] + sin(nu)',
                                    {'mu': (2,), 'nu': ()},
                                    'functional_with_derivative',
                                    dict_of_partial_derivatives)

mu = {'mu': (10,2), 'nu': 0}

derivative_to_first_mu_coordinate = epf.partial_derivative('mu', 0)
derivative_to_second_mu_coordinate = epf.partial_derivative('mu', 1)
derivative_to_nu_coordinate = epf.partial_derivative('nu', ())

der_mu_1 = derivative_to_first_mu_coordinate.evaluate(mu)
der_mu_2 = derivative_to_second_mu_coordinate.evaluate(mu)
der_nu = derivative_to_nu_coordinate.evaluate(mu)

assert epf.evaluate(mu) == 1004.
assert der_mu_1 == 100
assert der_mu_2 == 2
assert der_nu == 1


'''
test for mu_derivative of LincombOperator
'''

from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.basic import NumpyVectorSpace


space = NumpyVectorSpace(1)
zero_op = ZeroOperator(space, space)
operators = [zero_op, zero_op, zero_op]
coefficients = [1., pf, epf]

operator = LincombOperator(operators, coefficients)

op_sensitivity_to_first_mu = operator.mu_derivative('mu', (0,))
op_sensitivity_to_second_mu = operator.mu_derivative('mu', (1,))
op_sensitivity_to_nu = operator.mu_derivative('nu', ())

eval_mu_1 = op_sensitivity_to_first_mu.evaluate_coefficients(mu)
eval_mu_2 = op_sensitivity_to_second_mu.evaluate_coefficients(mu)
eval_nu = op_sensitivity_to_nu.evaluate_coefficients(mu)

assert operator.evaluate_coefficients(mu) == [1., 10, 1004.]
assert eval_mu_1 == [0., 1., 100.]
assert eval_mu_2 == [0., 0., 2.]
assert eval_nu == [0., 0., 1.]

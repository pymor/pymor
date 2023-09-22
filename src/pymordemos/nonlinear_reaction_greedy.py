from pymor.basic import *
from pymor.algorithms.newton import newton
from pymor.discretizers.builtin.cg import discretize_stationary_cg as discretizer
from pymor.analyticalproblems.elliptic import StationaryProblem
import numpy as np
import time
import sys

set_log_levels({'pymor': 'INFO'})


domain = RectDomain(([0,0], [1,1]))
l = ExpressionFunction('100 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', dim_domain = 2)
parameters = Parameters({'reaction': 2})
diffusion = ConstantFunction(1,2)

diameter = 1/40
this_mu = [10, 10]
ei_snapshots = 10
ei_size = 15
snapshots = 10
rb_size = 15
test = 5

nonlinear_reaction_coefficient = ConstantFunction(1,2)
test_nonlinearreaction = ExpressionFunction('reaction[0] * (exp(reaction[1] * u[0]) - 1) / reaction[1]', dim_domain = 1, parameters = parameters, variable = 'u')
test_nonlinearreaction_derivative = ExpressionFunction('reaction[0] * exp(reaction[1] * u[0])', dim_domain = 1, parameters = parameters, variable = 'u')
problem = StationaryProblem(domain = domain, rhs = l, diffusion = diffusion, nonlinear_reaction_coefficient = nonlinear_reaction_coefficient,
                                               nonlinear_reaction = test_nonlinearreaction, nonlinear_reaction_derivative = test_nonlinearreaction_derivative)
grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)
print('Anzahl Element', grid.size(0))
print('Anzahl DoFs', grid.size(2))
fom, data = discretizer(problem, diameter = diameter)

parameter_space = fom.parameters.space((0.01, 10))

pool = new_parallel_pool(ipython_num_engines=0, ipython_profile=None)
eim, ei_data = interpolate_operators(fom, ['operator'],
                                     parameter_space.sample_uniformly(ei_snapshots),
                                     error_norm=fom.h1_0_semi_norm, product=fom.h1_0_semi_product,
                                     max_interpolation_dofs=ei_size,
                                     alg='ei_greedy',
                                     pool=pool)

print('RB generation ...')

reductor = StationaryRBReductor(eim)

greedy_data = rb_greedy(fom, reductor, parameter_space.sample_uniformly(snapshots),
                        use_error_estimator=False, error_norm=lambda U: np.max(fom.h1_0_semi_norm(U)),
                        extension_params={'method': 'pod'}, max_extensions=rb_size,
                        pool=pool)

rom = greedy_data['rom']

mu = parameter_space.sample_randomly()
u_fom = fom.solve(mu)
u_rom = rom.solve(mu)
ERR = u_fom - reductor.reconstruct(u_rom)
print(f'rel. err.: {ERR.norm(fom.h1_0_semi_product)}')
fom.visualize(ERR)
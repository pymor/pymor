from pymor.basic import *
from pymor.algorithms.newton import newton
from pymor.discretizers.builtin.cg import discretize_stationary_cg as discretizer
from pymor.analyticalproblems.elliptic import StationaryProblem

set_log_levels({'pymor': 'INFO'})


domain = RectDomain(([0,0], [1,1]))
l = ExpressionFunction('100 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', dim_domain = 2)
parameters = Parameters({'reaction': 2})
diffusion = ConstantFunction(1,2)

diameter = 1/40
this_mu = [10, 10]

nonlinear_reaction_coefficient = ConstantFunction(1,2)
test_nonlinearreaction = ExpressionFunction('reaction[0] * (exp(reaction[1] * u[0]) - 1) / reaction[1]', dim_domain = 1, parameters = parameters, variable = 'u')
test_nonlinearreaction_derivative = ExpressionFunction('reaction[0] * exp(reaction[1] * u[0])', dim_domain = 1, parameters = parameters, variable = 'u')
problem = StationaryProblem(domain = domain, rhs = l, diffusion = diffusion, nonlinear_reaction_coefficient = nonlinear_reaction_coefficient,
                                               nonlinear_reaction = test_nonlinearreaction, nonlinear_reaction_derivative = test_nonlinearreaction_derivative)
grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)
print('Anzahl Element', grid.size(0))
print('Anzahl DoFs', grid.size(2))
fom, data = discretizer(problem, diameter = diameter)
# u = fom.solve(this_mu)
# u= newton(fom.operator, fom.rhs.as_range_array(), mu = problem.parameters.parse([0.01, 0.01]))[0]
# fom.visualize(u, title = 'cg')
# u= newton(fom.operator, fom.rhs.as_range_array(), mu = problem.parameters.parse([10, 10]))[0]
# fom.visualize(u, title = 'cg')
# problem_fv = StationaryProblem(domain = domain, rhs = l, diffusion = diffusion, nonlinear_reaction = test_nonlinearreaction, nonlinear_reaction_derivative = test_nonlinearreaction_derivative)
# fom_fv, data_fv = discretize_stationary_fv(problem_fv, diameter = diameter)
# u_fv = fom_fv.solve(this_mu)
# fom_fv.visualize(u_fv, title = 'fv')

parameter_space = fom.parameters.space((0.01, 10))

# ### ROM generation (POD/DEIM)
from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.newton import newton
from pymor.algorithms.pod import pod
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import StationaryRBReductor

U = fom.solution_space.empty()
residuals = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(10):
    UU, data = newton(fom.operator, fom.rhs.as_vector(), mu=mu, rtol=1e-6, return_residuals=True)
    U.append(UU)
    residuals.append(data['residuals'])

dofs, cb, _ = ei_greedy(residuals, rtol=1e-7)
ei_op = EmpiricalInterpolatedOperator(fom.operator, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)

rb, svals = pod(U, rtol=1e-7)
fom_ei = fom.with_(operator=ei_op)
reductor = StationaryRBReductor(fom_ei, rb)
rom = reductor.reduce()
# the reductor currently removes all solver_options so we need to add them again
rom = rom.with_(operator=rom.operator.with_(solver_options=fom.operator.solver_options))

# ### ROM validation
import time

import numpy as np

# ensure that FFC is not called during runtime measurements
rom.solve((1, 1))

errs = []
speedups = []
for mu in parameter_space.sample_randomly(10):
    tic = time.perf_counter()
    U = fom.solve(mu)
    t_fom = time.perf_counter() - tic

    tic = time.perf_counter()
    u_red = rom.solve(mu)
    t_rom = time.perf_counter() - tic

    U_red = reductor.reconstruct(u_red)
    errs.append(((U - U_red).norm() / U.norm())[0])
    speedups.append(t_fom / t_rom)
print(f'Maximum relative ROM error: {max(errs)}')
print(f'Median of ROM speedup: {np.median(speedups)}')
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Literal

from cyclopts import App

from pymor.analyticalproblems.domaindescriptions import LineDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv
from pymor.parameters.functionals import ProjectionParameterFunctional

app = App(help_on_error=True)

@app.default
def main(
    problem_number: Literal[0, 1],
    n: int,
    /, *,
    fv: bool = False,
):
    """Solves the Poisson equation in 1D using pyMOR's builtin discretization toolkit.

    Parameters
    ----------
    problem_number
        Selects the problem to solve [0 or 1].
    n
        Grid interval count.
    fv
        Use finite volume discretization instead of finite elements.
    """
    rhss = [ExpressionFunction('10', 1),
            ExpressionFunction('(x[0] - 0.5)**2 * 1000', 1)]
    rhs = rhss[problem_number]

    d0 = ExpressionFunction('1 - x[0]', 1)
    d1 = ExpressionFunction('x[0]', 1)

    f0 = ProjectionParameterFunctional('diffusionl')
    f1 = 1.

    problem = StationaryProblem(
        domain=LineDomain(),
        rhs=rhs,
        diffusion=LincombFunction([d0, d1], [f0, f1]),
        dirichlet_data=ConstantFunction(value=0, dim_domain=1),
        name='1DProblem'
    )

    parameter_space = problem.parameters.space(0.1, 1)

    print('Discretize ...')
    discretizer = discretize_stationary_fv if fv else discretize_stationary_cg
    m, data = discretizer(problem, diameter=1 / n)
    print(data['grid'])
    print()

    print('Solve ...')
    U = m.solution_space.empty()
    for mu in parameter_space.sample_uniformly(10):
        U.append(m.solve(mu))
    m.visualize(U, title='Solution for diffusionl in [0.1, 1]')


if __name__ == '__main__':
    app()

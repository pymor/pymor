#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys

import numpy as np
from typer import Argument, Option, run

from pymor.algorithms.greedy import rb_greedy
from pymor.algorithms.ei import interpolate_function
from pymor.algorithms.error import reduction_error_analysis
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor


def main(
    ei_snapshots: int = Argument(..., help='Number of snapshots for empirical interpolation.'),
    ei_size: int = Argument(..., help='Number of interpolation DOFs.'),
    snapshots: int = Argument(..., help='Number of snapshots for basis generation.'),
    rb_size: int = Argument(..., help='Size of the reduced basis.'),
    grid: int = Option(100, help='Use grid with 4*NI*NI elements'),
    plot_ei_err: bool = Option(False, help='Plot empirical interpolation error.'),
    plot_solutions: bool = Option(False, help='Plot some example solutions.'),
    test: int = Option(10, help='Number of snapshots to use for stochastic error estimation.'),
):
    """Reduction of a problem without parameter separability using empirical interpolation."""
    problem = StationaryProblem(
        domain=RectDomain(),

        diffusion=ExpressionFunction('(x[1] > 0.5) * x[0]**exponent[0] + 0.1', 2,
                                     parameters={'exponent': 1}),

        rhs=ConstantFunction(1., 2),

        parameter_ranges=(1, 10)
    )

    print('Discretize ...')
    fom, data = discretize_stationary_cg(problem, diameter=1. / grid)

    print(data['grid'])
    print(f'The parameters are {fom.parameters}')

    if plot_solutions:
        print('Showing some solutions')
        Us = ()
        legend = ()
        for mu in problem.parameter_space.sample_uniformly(4):
            print(f"Solving for exponent = {mu['exponent']} ... ")
            sys.stdout.flush()
            Us = Us + (fom.solve(mu),)
            legend = legend + (f"exponent: {mu['exponent']}",)
        fom.visualize(Us, legend=legend, title='Detailed Solutions', block=True)

    diffusion_ei, ei_data = interpolate_function(
        problem.diffusion,
        parameter_sample=problem.parameter_space.sample_uniformly(ei_snapshots),
        evaluation_points=data['grid'].centers(0),
        max_interpolation_dofs=ei_size
    )

    problem_ei = problem.with_(diffusion=diffusion_ei)
    fom_ei, _ = discretize_stationary_cg(problem_ei, diameter=1. / grid)

    if plot_ei_err:
        print('Showing some EI errors')
        ERRs = ()
        legend = ()
        for mu in problem.parameter_space.sample_randomly(2):
            print(f"Solving for exponent = \n{mu['exponent']} ... ")
            sys.stdout.flush()
            U = fom.solve(mu)
            U_EI = fom_ei.solve(mu)
            ERR = U - U_EI
            ERRs = ERRs + (ERR,)
            legend = legend + (f"exponent: {mu['exponent']}",)
            print(f'Error: {np.max(fom.l2_norm(ERR))}')
        fom.visualize(ERRs, legend=legend, title='EI Errors', separate_colorbars=True)

    print('RB generation ...')

    coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)
    reductor = CoerciveRBReductor(fom_ei, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator,
                                  check_orthonormality=False)

    greedy_data = rb_greedy(fom_ei, reductor, problem.parameter_space.sample_uniformly(snapshots),
                            max_extensions=rb_size)
    rom = greedy_data['rom']

    results = reduction_error_analysis(rom, fom=fom, reductor=reductor, error_estimator=True,
                                       error_norms=[fom.h1_0_semi_norm], condition=True,
                                       test_mus=problem.parameter_space.sample_randomly(test),
                                       plot=True)
    print(results['summary'])
    import matplotlib.pyplot
    matplotlib.pyplot.show()

    mumax = results['max_error_mus'][0, -1]
    U = fom.solve(mumax)
    U_RB = reductor.reconstruct(rom.solve(mumax))
    fom.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                  separate_colorbars=True, block=True)


if __name__ == '__main__':
    run(main)

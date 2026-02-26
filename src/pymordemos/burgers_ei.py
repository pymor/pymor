# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import math
import sys
import time
from typing import Literal

import numpy as np
from cyclopts import App

from pymor.algorithms.ei import interpolate_operators
from pymor.algorithms.greedy import rb_greedy
from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import RectGrid, TriaGrid, discretize_instationary_fv
from pymor.parallel.default import new_parallel_pool
from pymor.reductors.basic import InstationaryRBReductor

app = App(help_on_error=True)

@app.default
def main(
    exp_min: float,
    exp_max: float,
    ei_snapshots: int,
    ei_size: int,
    snapshots: int,
    rb_size: int,
    /, *,
    cache_region: Literal['none', 'memory', 'disk', 'persistent'] = 'disk',
    ei_alg: Literal['ei_greedy', 'deim', 'qdeim'] = 'ei_greedy',
    grid: int = 60,
    grid_type: Literal['rect', 'tria'] = 'rect',
    initial_data: Literal['sin', 'bump'] = 'sin',
    ipython_engines: int = 0,
    ipython_profile: str | None = None,
    lxf_lambda: float = 1.,
    periodic: bool = True,
    nt: int = 100,
    num_flux: Literal['lax_friedrichs', 'engquist_osher'] = 'engquist_osher',
    plot_err: bool = False,
    plot_ei_err: bool = False,
    plot_error_landscape: bool = False,
    plot_error_landscape_M: int = 10,
    plot_error_landscape_N: int = 10,
    plot_solutions: bool = False,
    test: int = 10,
    vx: float = 1.,
    vy: float = 1.
):
    """Reduction of a two-dimensional Burgers-type equation using empirical interpolation.

    Parameters
    ----------
    exp_min
        Minimal exponent.
    exp_max
        Maximal exponent.
    ei_snapshots
        Number of snapshots for empirical interpolation.
    ei_size
        Number of interpolation DOFs.
    snapshots
        Number of snapshots for basis generation.
    rb_size
        Size of the reduced basis.
    cache_region
        Name of cache region to use for caching solution snapshots.
    ei_alg
        Interpolation algorithm to use.
    grid
        Use grid with (2*NI)*NI elements.
    grid_type
        Type of grid to use.
    initial_data
        Select the initial data (sin, bump).
    ipython_engines:
        If positive, the number of IPython cluster engines to use for parallel greedy search.
        If zero, no parallelization is performed.
    ipython_profile
        IPython profile to use for parallelization.
    lxf_lambda
        Parameter lambda in Lax-Friedrichs flux.
    periodic
        If not, solve with dirichlet boundary conditions on left and bottom boundary.
    nt
        Number of time steps.
    num_flux
        Numerical flux to use.
    plot_err
        Plot error.
    plot_ei_err
        Plot empirical interpolation error.
    plot_error_landscape
        Calculate and show plot of reduction error vs. basis sizes.
    plot_error_landscape_M
        Number of collateral basis sizes to test.
    plot_error_landscape_N
        Number of basis sizes to test.
    plot_solutions
        Plot some example solutions.
    test
        Number of snapshots to use for stochastic error estimation.
    vx
        Speed in x-direction.
    vy
        Speed in y-direction.
    """
    print('Setup Problem ...')
    problem = burgers_problem_2d(vx=vx, vy=vy, initial_data_type=initial_data,
                                 parameter_range=(exp_min, exp_max), torus=periodic)

    print('Discretize ...')
    if grid_type == 'rect':
        grid *= 1. / math.sqrt(2)
    fom, _ = discretize_instationary_fv(
        problem,
        diameter=1. / grid,
        grid_type=RectGrid if grid_type == 'rect' else TriaGrid,
        num_flux=num_flux,
        lxf_lambda=lxf_lambda,
        nt=nt
    )

    if cache_region != 'none':
        # building a cache_id is only needed for persistent CacheRegions
        cache_id = (f'pymordemos.burgers_ei {vx} {vy} {initial_data}'
                    f'{periodic} {grid} {grid_type} {num_flux} {lxf_lambda} {nt}')
        fom.enable_caching(cache_region, cache_id)

    print(fom.operator.grid)

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

    pool = new_parallel_pool(ipython_num_engines=ipython_engines, ipython_profile=ipython_profile)
    eim, ei_data = interpolate_operators(fom, ['operator'],
                                         problem.parameter_space.sample_uniformly(ei_snapshots),
                                         error_norm=fom.l2_norm, product=fom.l2_product,
                                         max_interpolation_dofs=ei_size,
                                         alg=ei_alg,
                                         pool=pool)

    if plot_ei_err:
        print('Showing some EI errors')
        ERRs = ()
        legend = ()
        for mu in problem.parameter_space.sample_randomly(2):
            print(f"Solving for exponent = \n{mu['exponent']} ... ")
            sys.stdout.flush()
            U = fom.solve(mu)
            U_EI = eim.solve(mu)
            ERR = U - U_EI
            ERRs = ERRs + (ERR,)
            legend = legend + (f"exponent: {mu['exponent']}",)
            print(f'Error: {np.max(fom.l2_norm(ERR))}')
        fom.visualize(ERRs, legend=legend, title='EI Errors', separate_colorbars=True)

        print('Showing interpolation DOFs ...')
        U = np.zeros(U.dim)
        dofs = eim.operator.interpolation_dofs
        U[dofs] = np.arange(1, len(dofs) + 1)
        U[eim.operator.source_dofs] += int(len(dofs)/2)
        fom.visualize(fom.solution_space.make_array(U),
                      title='Interpolation DOFs')

    print('RB generation ...')

    reductor = InstationaryRBReductor(eim)

    greedy_data = rb_greedy(fom, reductor, problem.parameter_space.sample_uniformly(snapshots),
                            use_error_estimator=False, error_norm=lambda U: np.max(fom.l2_norm(U)),
                            extension_params={'method': 'pod'}, max_extensions=rb_size,
                            pool=pool)

    rom = greedy_data['rom']

    print('\nSearching for maximum error on random snapshots ...')

    tic = time.perf_counter()

    mus = problem.parameter_space.sample_randomly(test)

    def error_analysis(N, M):
        print(f'N = {N}, M = {M}: ', end='')
        rom = reductor.reduce(N)
        rom = rom.with_(operator=rom.operator.with_cb_dim(M))
        l2_err_max = -1
        mumax = None
        for mu in mus:
            print('.', end='')
            sys.stdout.flush()
            u = rom.solve(mu)
            URB = reductor.reconstruct(u)
            U = fom.solve(mu)
            l2_err = np.max(fom.l2_norm(U - URB))
            l2_err = np.inf if not np.isfinite(l2_err) else l2_err
            if l2_err > l2_err_max:
                l2_err_max = l2_err
                mumax = mu
        print()
        return l2_err_max, mumax
    error_analysis = np.frompyfunc(error_analysis, 2, 2)

    real_rb_size = len(reductor.bases['RB'])
    real_cb_size = len(ei_data['basis'])
    if plot_error_landscape:
        N_count = min(real_rb_size - 1, plot_error_landscape_N)
        M_count = min(real_cb_size - 1, plot_error_landscape_M)
        Ns = np.linspace(1, real_rb_size, N_count).astype(int)
        Ms = np.linspace(1, real_cb_size, M_count).astype(int)
    else:
        Ns = np.array([real_rb_size])
        Ms = np.array([real_cb_size])

    N_grid, M_grid = np.meshgrid(Ns, Ms)

    errs, err_mus = error_analysis(N_grid, M_grid)
    errs = errs.astype(np.float64)

    l2_err_max = errs[-1, -1]
    mumax = err_mus[-1, -1]
    toc = time.perf_counter()
    t_est = toc - tic

    print("""
    *** RESULTS ***

    Problem:
       parameter range:                    ({exp_min}, {exp_max})
       h:                                  sqrt(2)/{grid}
       grid-type:                          {grid_type}
       initial-data:                       {initial_data}
       lxf-lambda:                         {lxf_lambda}
       nt:                                 {nt}
       not-periodic:                       {periodic}
       num-flux:                           {num_flux}
       (vx, vy):                           ({vx}, {vy})

    Greedy basis generation:
       number of ei-snapshots:             {ei_snapshots}
       prescribed collateral basis size:   {ei_size}
       actual collateral basis size:       {real_cb_size}
       number of snapshots:                {snapshots}
       prescribed basis size:              {rb_size}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {test}
       maximal L2-error:                   {l2_err_max}  (mu = {mumax})
       elapsed time:                       {t_est}
    """.format(**locals()))

    sys.stdout.flush()
    if plot_error_landscape:
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # rescale the errors since matplotlib does not support logarithmic scales on 3d plots
        # https://github.com/matplotlib/matplotlib/issues/209
        surf = ax.plot_surface(M_grid, N_grid, np.log(np.minimum(errs, 1)) / np.log(10),
                               rstride=1, cstride=1, cmap='jet')
        plt.show()
    if plot_err:
        U = fom.solve(mumax)
        URB = reductor.reconstruct(rom.solve(mumax))
        fom.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                      title='Maximum Error Solution', separate_colorbars=True)

    global test_results
    test_results = (ei_data, greedy_data)


if __name__ == '__main__':
    app()

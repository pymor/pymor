# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np
from typer import Argument, run

from pymor.algorithms.pod import pod
from pymor.analyticalproblems.functions import ExpressionFunction
from pymor.core.config import config
from pymor.models.examples import stokes_2Dexample
from pymor.reductors.stokes import StationaryLSRBStokesReductor, StationarySupremizerGalerkinStokesReductor

PROJECTION_METHODS = ['supremizer_galerkin', 'ls-normal', 'ls-ls']

def main(
    mu_low: float = Argument(0.01),
    mu_high: float = Argument(1000),
    modes: int = Argument(50),
    n_tests: int = Argument(10)
):
    """This example sets up the MOR workflow for the 2D stationary Stokes equation.

    The script first computes POD bases for the velocity and pressure spaces
    from randomly sampled solutions of the full order model (FOM). Then, it
    constructs reduced order models (ROMs) using three different projection
    methods: supremizer enrichment Galerkin projection, least-squares projection with
    normal equations, and least-squares projection without normal equations.  Finally,
    it evaluates the ROMs against the FOM on a set of random parameters.
    """
    # sets up the discrete Stokes model
    config.require('SCIKIT_FEM')
    body_force = ExpressionFunction(('[0, x[0]]'), dim_domain=2)
    fom_stokes = stokes_2Dexample(rhs=body_force)

    # compute snapshot for the pressure and the velocity space
    snapshots_u = fom_stokes.solution_space.subspaces[0].empty()
    snapshots_p = fom_stokes.solution_space.subspaces[1].empty()

    for i in range(modes):
        rng = np.random.default_rng(i)
        mu = {'mu': rng.uniform(mu_low, mu_high)}
        sol_u, sol_p = fom_stokes.solve(mu).blocks
        snapshots_u.append(sol_u)
        snapshots_p.append(sol_p)

    basis_u, _ = pod(snapshots_u, modes=modes, rtol=1e-17, orth_tol=1e-17, product=fom_stokes.u_product)
    basis_p, _ = pod(snapshots_p, modes=modes, rtol=1e-17, orth_tol=1e-17, product=fom_stokes.p_product)

    speedups = {}
    errors_u = {}
    errors_p = {}

    rng = np.random.default_rng(442)
    mus = rng.uniform(mu_low, mu_high, n_tests).tolist()
    results_fom = [evaluate_fom_once(fom_stokes, mu) for mu in mus]

    for method in PROJECTION_METHODS:

        # construct reduced order model
        if method == 'supremizer_galerkin':
            reductor_stokes = StationarySupremizerGalerkinStokesReductor(fom_stokes,
                                                                         RB_u=basis_u,
                                                                         RB_p=basis_p,
                                                                         u_product = fom_stokes.u_product,
                                                                         p_product=fom_stokes.p_product)
        elif method == 'ls-normal':
            reductor_stokes = StationaryLSRBStokesReductor(fom_stokes,
                                                           RB_u=basis_u,
                                                           RB_p=basis_p,
                                                           u_product = fom_stokes.u_product,
                                                           p_product=fom_stokes.p_product,
                                                           use_normal_equations=True)
        elif method == 'ls-ls':
            reductor_stokes = StationaryLSRBStokesReductor(fom_stokes,
                                                           RB_u=basis_u,
                                                           RB_p=basis_p,
                                                           u_product = fom_stokes.u_product,
                                                           p_product=fom_stokes.p_product,
                                                           use_normal_equations=False)
        else:
            raise ValueError(f'Unknown projection method {method}')

        rom = reductor_stokes.reduce()
        results_rom = [evaluate_rom_once(rom, reductor_stokes, mu) for mu in mus]
        results = compute_speedup_and_errros(results_fom, results_rom)

        speedups[method] = np.mean([r['speedup'] for r in results])
        errors_u[method] = np.mean([r['err_u'] for r in results])
        errors_p[method] = np.mean([r['err_p'] for r in results])

    print_results(speedups, errors_u, errors_p)

def evaluate_fom_once(fom, mu):
    # FOM solve & timing
    tic_fom = time.perf_counter()
    fom_u, fom_p = fom.solve(mu).blocks
    t_fom = time.perf_counter() - tic_fom

    return {'fom_u': fom_u, 'fom_p': fom_p, 't_fom': t_fom}

def evaluate_rom_once(rom, reductor_stokes, mu):
    # ROM solve & timing
    tic_rom = time.perf_counter()
    test_sol_rom = rom.solve(mu)
    t_rom = time.perf_counter() - tic_rom

    test_sol_rom_re = reductor_stokes.reconstruct(test_sol_rom)
    return {'rom': test_sol_rom_re,'t_rom': t_rom}

def compute_speedup_and_errros(fom_results, rom_results):
    results = []

    for i in range(len(fom_results)):
        # Relative errors
        results.append({})
        dim_u = fom_results[i]['fom_u'].to_numpy().shape[0]
        dim_p = fom_results[i]['fom_p'].to_numpy().shape[0]
        results[i]['err_u'] = np.linalg.norm(rom_results[i]['rom'].dofs(list(range(dim_u))) -
                                             fom_results[i]['fom_u'].to_numpy()) / fom_results[i]['fom_u'].norm()
        results[i]['err_p'] = np.linalg.norm(rom_results[i]['rom'].dofs(list(range(dim_u, dim_u + dim_p))) -
                                             fom_results[i]['fom_p'].to_numpy()) / fom_results[i]['fom_p'].norm()

        # Speedup (FOM time divided by ROM time)
        results[i]['speedup'] = (fom_results[i]['t_fom'] / rom_results[i]['t_rom'])

    return results

def print_results(speedups, errors_u, errors_p):
    print('\n======== ROM Evaluation Results ========\n')
    print(f"{'Method':<20} | {'Speedup':>10} | {'Error u':>12} | {'Error p':>12}")
    print('-' * 56)
    for method in speedups:
        print(f'{method:<20} | {speedups[method]:10.2f} | {errors_u[method]:12.4e} | {errors_p[method]:12.4e}')
    print()

if __name__ == '__main__':
    run(main)

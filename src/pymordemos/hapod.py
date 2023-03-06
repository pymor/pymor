#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from typer import Argument, Option, run

from pymor.algorithms.hapod import dist_vectorarray_hapod, inc_vectorarray_hapod
from pymor.algorithms.pod import pod
from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import RectGrid, discretize_instationary_fv
from pymor.tools.table import format_table


def main(
    tol: float = Argument(..., help='Prescribed mean l2 approximation error.'),
    dist: int = Argument(..., help='Number of slices for distributed HAPOD.'),
    inc: int = Argument(..., help='Number of steps for incremental HAPOD.'),

    arity: int = Option(None, help='Arity of distributed HAPOD tree'),
    grid: int = Option(60, help='Use grid with (2*NI)*NI elements.'),
    nt: int = Option(100, help='Number of time steps.'),
    omega: float = Option(0.9, help='Parameter omega from HAPOD algorithm.'),
    procs: int = Option(0, help='Number of processes to use for parallelization.'),
    snap: int = Option(20, help='Number of snapshot trajectories to compute.'),
    threads: int = Option(0, help='Number of threads to use for parallelization.'),
    svd: bool = Option(False, help='Compute SVD.'),
):
    """Compression of snapshot data with the HAPOD algorithm from [HLR18]."""
    assert procs == 0 or threads == 0

    executor = ProcessPoolExecutor(procs) if procs > 0 else \
        ThreadPoolExecutor(threads) if threads > 0 else \
        None

    p = burgers_problem_2d()
    m, data = discretize_instationary_fv(p, grid_type=RectGrid, diameter=np.sqrt(2)/grid, nt=nt)

    U = m.solution_space.empty()
    for mu in p.parameter_space.sample_randomly(snap):
        U.append(m.solve(mu))

    tic = time.perf_counter()
    if svd:
        pod_modes, pod_svals, pod_rmodes = pod(U, l2_err=tol * np.sqrt(len(U)), product=m.l2_product,
                                               return_right_singular_vectors=True)
    else:
        pod_modes = pod(U, l2_err=tol * np.sqrt(len(U)), product=m.l2_product)[0]
    pod_time = time.perf_counter() - tic

    tic = time.perf_counter()
    if svd:
        dist_modes, dist_svals, dist_rmodes = \
            dist_vectorarray_hapod(dist, U, tol, omega, arity=arity, product=m.l2_product, executor=executor,
                                   return_right_singular_vectors=True)[0:3]
    else:
        dist_modes = \
            dist_vectorarray_hapod(dist, U, tol, omega, arity=arity, product=m.l2_product, executor=executor)[0]
    dist_time = time.perf_counter() - tic

    tic = time.perf_counter()
    if svd:
        inc_modes, inc_svals, inc_rmodes = \
            inc_vectorarray_hapod(inc, U, tol, omega, product=m.l2_product,
                                  return_right_singular_vectors=True)[0:3]
    else:
        inc_modes = inc_vectorarray_hapod(inc, U, tol, omega, product=m.l2_product)[0]
    inc_time = time.perf_counter() - tic

    if svd:
        pod_err = np.linalg.norm(m.l2_norm(U-pod_modes.lincomb(pod_svals * pod_rmodes.T))/np.sqrt(len(U)))
        dist_err = np.linalg.norm(m.l2_norm(U-dist_modes.lincomb(dist_svals * dist_rmodes.T))/np.sqrt(len(U)))
        inc_err = np.linalg.norm(m.l2_norm(U-inc_modes.lincomb(inc_svals * inc_rmodes.T))/np.sqrt(len(U)))
    else:
        pod_err = np.linalg.norm(m.l2_norm(U-pod_modes.lincomb(m.l2_product.apply2(U, pod_modes)))/np.sqrt(len(U)))
        dist_err = np.linalg.norm(m.l2_norm(U-dist_modes.lincomb(m.l2_product.apply2(U, dist_modes)))/np.sqrt(len(U)))
        inc_err = np.linalg.norm(m.l2_norm(U-inc_modes.lincomb(m.l2_product.apply2(U, inc_modes)))/np.sqrt(len(U)))

    print(f'Snapshot matrix: {U.dim} x {len(U)}')
    print(format_table([
        ['Method', 'Error', 'Modes', 'Time'],
        ['POD',
         pod_err,
         len(pod_modes),
         pod_time],
        ['DIST HAPOD',
         dist_err,
         len(dist_modes),
         dist_time],
        ['INC HAPOD',
         inc_err,
         len(inc_modes),
         inc_time]]
    ))


if __name__ == '__main__':
    run(main)

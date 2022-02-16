#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

import numpy as np
from typer import Argument, Option, run

from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import discretize_instationary_fv, RectGrid
from pymor.algorithms.hapod import dist_vectorarray_hapod, inc_vectorarray_hapod
from pymor.algorithms.pod import pod
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
    pod_modes = pod(U, l2_err=tol * np.sqrt(len(U)), product=m.l2_product)[0]
    pod_time = time.perf_counter() - tic

    tic = time.perf_counter()
    dist_modes = dist_vectorarray_hapod(dist, U, tol, omega, arity=arity, product=m.l2_product, executor=executor)[0]
    dist_time = time.perf_counter() - tic

    tic = time.perf_counter()
    inc_modes = inc_vectorarray_hapod(inc, U, tol, omega, product=m.l2_product)[0]
    inc_time = time.perf_counter() - tic

    print(f'Snapshot matrix: {U.dim} x {len(U)}')
    print(format_table([
        ['Method', 'Error', 'Modes', 'Time'],
        ['POD',
         np.linalg.norm(m.l2_norm(U-pod_modes.lincomb(m.l2_product.apply2(U, pod_modes)))/np.sqrt(len(U))),
         len(pod_modes),
         pod_time],
        ['DIST HAPOD',
         np.linalg.norm(m.l2_norm(U-dist_modes.lincomb(m.l2_product.apply2(U, dist_modes)))/np.sqrt(len(U))),
         len(dist_modes),
         dist_time],
        ['INC HAPOD',
         np.linalg.norm(m.l2_norm(U-inc_modes.lincomb(m.l2_product.apply2(U, inc_modes)))/np.sqrt(len(U))),
         len(inc_modes),
         inc_time]]
    ))


if __name__ == '__main__':
    run(main)

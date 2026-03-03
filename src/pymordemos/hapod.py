# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from cyclopts import App

from pymor.algorithms.hapod import dist_vectorarray_hapod, inc_vectorarray_hapod
from pymor.algorithms.pod import pod
from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import RectGrid, discretize_instationary_fv
from pymor.tools.table import format_table

app = App(help_on_error=True)

@app.default
def main(
    tol: float,
    dist: int,
    inc: int,
    /, *,
    arity: int | None = None,
    grid: int = 60,
    nt: int = 100,
    omega: float = 0.9,
    procs: int = 0,
    snap: int = 20,
    threads: int = 0
):
    """Compression of snapshot data with the HAPOD algorithm from [HLR18].

    Parameters
    ----------
    tol
        Prescribed mean l2 approximation error.
    dist
        Number of slices for distributed HAPOD.
    inc
        Number of steps for incremental HAPOD.
    arity
        Arity of distributed HAPOD tree
    grid
        Use grid with (2*NI)*NI elements.
    nt
        Number of time steps.
    omega
        Parameter omega from HAPOD algorithm.
    procs
        Number of processes to use for parallelization.
    snap
        Number of snapshot trajectories to compute.
    threads
        Number of threads to use for parallelization.
    """
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
         np.linalg.norm(m.l2_norm(U-pod_modes.lincomb(m.l2_product.apply2(pod_modes, U)))/np.sqrt(len(U))),
         len(pod_modes),
         pod_time],
        ['DIST HAPOD',
         np.linalg.norm(m.l2_norm(U-dist_modes.lincomb(m.l2_product.apply2(dist_modes, U)))/np.sqrt(len(U))),
         len(dist_modes),
         dist_time],
        ['INC HAPOD',
         np.linalg.norm(m.l2_norm(U-inc_modes.lincomb(m.l2_product.apply2(inc_modes, U)))/np.sqrt(len(U))),
         len(inc_modes),
         inc_time]]
    ))


if __name__ == '__main__':
    app()

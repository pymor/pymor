#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import Argument, run

from pymor.algorithms.pod import pod
from pymor.algorithms.symplectic import psd_complex_svd, psd_cotengent_lift, psd_svd_like_decomp
from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import IdentityOperator, LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import InstationaryRBReductor
from pymor.reductors.symplectic import QuadraticHamiltonianRBReductor
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
from scipy.sparse import diags

SYMPLECTIC_METHODS = ['cotangent_lift', 'complex_svd', 'svd_like']
METHODS = ['pod'] + SYMPLECTIC_METHODS


def main(
    final_time: float = Argument(10., help='Final time of the simulation'),
    rbsize: int = Argument(80, help='Maximal reduced basis size'),
):
    """Symplectic MOR experiment for linear wave equation discretized with FD.

    The experiment closely follows the experiment described in :cite:`PM16`. The reduced models are
    trained on the trajectory of one parameter and try to reproduce this solution in the reduced
    simulation (reproduction experiment).

    It compares structure-preserving MOR for Hamiltonian systems (known as symplectic MOR) with
    classical (non-structure-preserving) MOR. Different symplectic basis generation techniques are
    compared ('cotangent_lift', 'complex_svd', 'svd_like') to a non-symplectic basis ('pod').
    The experiment shows: Although 'pod' has the best projection error, its reduction error is
    comparably high. In contrast to this, the reduction error of all symplectic bases is close to
    their respective projection error.

    Note that compared to the experiments in :cite:`PM16`, the POD gives better results here.
    """
    from matplotlib import pyplot as plt

    # deactivate warnings about missing solver_options {'type': 'to_matrix'}
    from pymor.core.logger import set_log_levels
    set_log_levels({'pymor.operators.block.BlockOperator': 'ERROR'})

    # compute errors for reproduction experiment
    fom = discretize_fom(T=final_time)
    U_fom = fom.solve()
    rel_fac = np.sqrt(U_fom.norm2().sum())

    # run mor for all METHODS
    half_rbsize = min(rbsize // 2, len(U_fom) // 2)
    red_dims = np.linspace(0, half_rbsize, 10, dtype=int) * 2
    results = {}
    for method in METHODS:
        results[method] = run_mor(fom, U_fom, method, red_dims)

    # plot results
    markers = {
        'pod': '^',
        'cotangent_lift': '.',
        'complex_svd': 's',
        'svd_like': 'X',
    }
    colors = {
        'pod': 'blue',
        'cotangent_lift': 'red',
        'complex_svd': 'green',
        'svd_like': 'gray',
    }
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
    for method, result in results.items():
        axs[0].semilogy(
            red_dims,
            result['abs_err_proj'] / rel_fac,
            marker=markers[method],
            color=colors[method])
        axs[1].semilogy(
            red_dims,
            result['abs_err_rom'] / rel_fac,
            label=method,
            marker=markers[method],
            color=colors[method])

    fig.suptitle('Linear wave equation, FD discretization, reproduction experiment')
    axs[0].title.set_text('Relative projection error')
    axs[1].title.set_text('Relative reduction error')
    axs[0].set_xlabel('red. dim. 2k')
    axs[1].set_xlabel('red. dim. 2k')
    axs[0].set_ylabel('rel. err.')

    plt.legend()
    plt.show()


def discretize_fom(T=50):
    n_x = 500
    nt = int(T / 0.01) + 1
    wave_speed = 0.1
    l = 1.
    dx = l / (n_x-1)

    # construct H_op
    space = NumpyVectorSpace(n_x)
    Dxx = diags(
        [-2 * np.ones(n_x), np.ones(n_x-1), np.ones(n_x-1)],
        [0, 1, -1],
        format='csr',
    )
    H_op = BlockDiagonalOperator([
        NumpyMatrixOperator(-wave_speed**2 / dx * Dxx),
        LincombOperator([IdentityOperator(space)], [1/dx]),
    ])

    # construct initial_data
    h = lambda s: (0 <= s) * (s <= 1) * (1 - 3/2 * s**2 + 3/4 * s**3) + (1 < s) * (s <= 2) * ((2-s)**3)/4
    bump = lambda xi: h(np.abs(4*(xi - l/2)))
    initial_data = H_op.source.make_array([
        space.make_array(bump(np.linspace(0, l, n_x))),
        space.make_array(np.zeros(n_x)),
    ])

    fom = QuadraticHamiltonianModel(T, initial_data, H_op, nt=nt, name='hamiltonian_wave_equation')
    # fom.operator = fom.operator.with_(solver_options={'type': 'to_matrix'}) #TODO
    return fom


def run_mor(fom, U_fom, method, red_dims):
    assert isinstance(fom, QuadraticHamiltonianModel)
    assert isinstance(U_fom, VectorArray) and U_fom in fom.H_op.range
    assert isinstance(method, str) and method in METHODS
    assert isinstance(red_dims, np.ndarray) and red_dims.dtype == int
    assert fom.time_stepper.nt + 1 == len(U_fom)

    # compute basis of maximal size
    max_red_dim = red_dims.max()
    if method in SYMPLECTIC_METHODS:
        if method == 'cotangent_lift':
            MAX_RB = psd_cotengent_lift(U_fom, max_red_dim)
        elif method == 'complex_svd':
            MAX_RB = psd_complex_svd(U_fom, max_red_dim)
        elif method == 'svd_like':
            MAX_RB = psd_svd_like_decomp(U_fom, max_red_dim)
        else:
            raise NotImplementedError('Unknown method: {}'.format(method))
    else:
        assert method == 'pod'
        MAX_RB, svals = pod(U_fom, modes=max_red_dim)

    # compute ROM results for all reduced dimensions
    abs_err_proj = np.zeros(len(red_dims))
    abs_err_rom = np.zeros(len(red_dims))
    for i_red_dim, red_dim in enumerate(red_dims):
        if red_dim > len(MAX_RB) * (2 if method in SYMPLECTIC_METHODS else 1):
            abs_err_proj[i_red_dim] = np.nan
            abs_err_rom[i_red_dim] = np.nan
            continue
        if method in SYMPLECTIC_METHODS:
            RB = MAX_RB[:red_dim//2]
            reductor = QuadraticHamiltonianRBReductor(fom, RB)
            RB_tsi = RB.transposed_symplectic_inverse()
            U_proj = RB.lincomb(U_fom.inner(RB_tsi.to_array()))
        else:
            RB = MAX_RB[:red_dim]
            reductor = InstationaryRBReductor(fom, RB)
            U_proj = RB.lincomb(U_fom.inner(RB))
        rom = reductor.reduce()
        u = rom.solve()
        abs_err_proj[i_red_dim] = np.sqrt((U_fom - U_proj).norm2().sum())
        abs_err_rom[i_red_dim] = np.sqrt((U_fom - reductor.reconstruct(u)).norm2().sum())
    return {
        'abs_err_proj': abs_err_proj,
        'abs_err_rom': abs_err_rom,
    }


if __name__ == '__main__':
    run(main)

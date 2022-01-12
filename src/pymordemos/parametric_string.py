#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.core.config import config
from pymor.core.logger import set_log_levels
from pymor.models.iosys import SecondOrderModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.mt import MTReductor
from pymor.reductors.sobt import (SOBTpReductor, SOBTvReductor, SOBTpvReductor, SOBTvpReductor,
                                  SOBTfvReductor, SOBTReductor)
from pymor.reductors.sor_irka import SORIRKAReductor
from pymordemos.parametric_heat import run_mor_method_param


def main(
        n: int = Argument(101, help='Order of the full second-order model (odd number).'),
        r: int = Argument(5, help='Order of the ROMs.'),
):
    """Parametric string example."""
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

    # Assemble M, D, K, B, C_p
    assert n % 2 == 1, 'The order has to be an odd integer.'

    n2 = (n + 1) // 2

    k = 0.01  # stiffness

    M = sps.eye(n, format='csc')
    E = sps.eye(n, format='csc')
    K = sps.diags([n * [2 * k * n ** 2],
                   (n - 1) * [-k * n ** 2],
                   (n - 1) * [-k * n ** 2]],
                  [0, -1, 1],
                  format='csc')
    B = np.zeros((n, 1))
    B[n2 - 1, 0] = n
    Cp = np.zeros((1, n))
    Cp[0, n2 - 1] = 1

    # Second-order system
    Mop = NumpyMatrixOperator(M)
    Eop = NumpyMatrixOperator(E) * ProjectionParameterFunctional('damping')
    Kop = NumpyMatrixOperator(K)
    Bop = NumpyMatrixOperator(B)
    Cpop = NumpyMatrixOperator(Cp)

    so_sys = SecondOrderModel(Mop, Eop, Kop, Bop, Cpop)

    print(f'order of the model = {so_sys.order}')
    print(f'number of inputs   = {so_sys.dim_input}')
    print(f'number of outputs  = {so_sys.dim_output}')

    mu_list = [1, 5, 10]
    w = np.logspace(-3, 2, 200)

    # System poles
    fig, ax = plt.subplots()
    for mu in mu_list:
        poles = so_sys.poles(mu=mu)
        ax.plot(poles.real, poles.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title('System poles')
    ax.legend()
    plt.show()

    # Magnitude plots
    fig, ax = plt.subplots()
    for mu in mu_list:
        so_sys.transfer_function.mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the full model')
    ax.legend()
    plt.show()

    # "Hankel" singular values
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    for mu in mu_list:
        psv = so_sys.psv(mu=mu)
        vsv = so_sys.vsv(mu=mu)
        pvsv = so_sys.pvsv(mu=mu)
        vpsv = so_sys.vpsv(mu=mu)
        ax[0, 0].semilogy(range(1, len(psv) + 1), psv, '.-', label=fr'$\mu = {mu}$')
        ax[0, 1].semilogy(range(1, len(vsv) + 1), vsv, '.-')
        ax[1, 0].semilogy(range(1, len(pvsv) + 1), pvsv, '.-')
        ax[1, 1].semilogy(range(1, len(vpsv) + 1), vpsv, '.-')
    ax[0, 0].set_title('Position singular values')
    ax[0, 1].set_title('Velocity singular values')
    ax[1, 0].set_title('Position-velocity singular values')
    ax[1, 1].set_title('Velocity-position singular values')
    fig.legend(loc='upper center', ncol=len(mu_list))
    plt.show()

    # System norms
    for mu in mu_list:
        print(f'mu = {mu}:')
        print(f'    H_2-norm of the full model:    {so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    H_inf-norm of the full model:  {so_sys.hinf_norm(mu=mu):e}')
        print(f'    Hankel-norm of the full model: {so_sys.hankel_norm(mu=mu):e}')

    # Model order reduction
    run_mor_method_param(so_sys, r, w, mu_list, SOBTpReductor, 'SOBTp')
    run_mor_method_param(so_sys, r, w, mu_list, SOBTvReductor, 'SOBTv')
    run_mor_method_param(so_sys, r, w, mu_list, SOBTpvReductor, 'SOBTpv')
    run_mor_method_param(so_sys, r, w, mu_list, SOBTvpReductor, 'SOBTvp')
    run_mor_method_param(so_sys, r, w, mu_list, SOBTfvReductor, 'SOBTfv')
    run_mor_method_param(so_sys, r, w, mu_list, SOBTReductor, 'SOBT')
    run_mor_method_param(so_sys, r, w, mu_list, SORIRKAReductor, 'SOR-IRKA')
    run_mor_method_param(so_sys.to_lti(), r, w, mu_list, BTReductor, 'BT')
    run_mor_method_param(so_sys.to_lti(), r, w, mu_list, IRKAReductor, 'IRKA')
    run_mor_method_param(so_sys.to_lti(), r, w, mu_list, MTReductor, 'MT')


if __name__ == "__main__":
    run(main)

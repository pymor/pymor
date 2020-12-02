#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from typer import Argument, run

from pymor.core.config import config
from pymor.core.logger import set_log_levels
from pymor.models.iosys import SecondOrderModel
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.sobt import (SOBTpReductor, SOBTvReductor, SOBTpvReductor, SOBTvpReductor,
                                  SOBTfvReductor, SOBTReductor)
from pymor.reductors.sor_irka import SORIRKAReductor
from pymordemos.heat import run_mor_method


def main(
        n: int = Argument(101, help='Order of the full second-order model (odd number).'),
        r: int = Argument(5, help='Order of the ROMs.'),
):
    """String equation example."""
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'ERROR'})

    # Assemble matrices
    assert n % 2 == 1, 'The order has to be an odd integer.'

    n2 = (n + 1) // 2

    d = 10  # damping
    k = 0.01   # stiffness

    M = sps.eye(n, format='csc')
    E = d * sps.eye(n, format='csc')
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
    so_sys = SecondOrderModel.from_matrices(M, E, K, B, Cp)

    print(f'order of the model = {so_sys.order}')
    print(f'number of inputs   = {so_sys.dim_input}')
    print(f'number of outputs  = {so_sys.dim_output}')

    poles = so_sys.poles()
    fig, ax = plt.subplots()
    ax.plot(poles.real, poles.imag, '.')
    ax.set_title('System poles')
    plt.show()

    w = np.logspace(-4, 2, 200)
    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    ax.set_title('Magnitude plot of the full model')
    plt.show()

    psv = so_sys.psv()
    vsv = so_sys.vsv()
    pvsv = so_sys.pvsv()
    vpsv = so_sys.vpsv()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    ax[0, 0].semilogy(range(1, len(psv) + 1), psv, '.-')
    ax[0, 0].set_title('Position singular values')
    ax[0, 1].semilogy(range(1, len(vsv) + 1), vsv, '.-')
    ax[0, 1].set_title('Velocity singular values')
    ax[1, 0].semilogy(range(1, len(pvsv) + 1), pvsv, '.-')
    ax[1, 0].set_title('Position-velocity singular values')
    ax[1, 1].semilogy(range(1, len(vpsv) + 1), vpsv, '.-')
    ax[1, 1].set_title('Velocity-position singular values')
    plt.show()

    print(f'FOM H_2-norm:    {so_sys.h2_norm():e}')
    if config.HAVE_SLYCOT:
        print(f'FOM H_inf-norm:  {so_sys.hinf_norm():e}')
    else:
        print('H_inf-norm calculation is skipped due to missing slycot.')
    print(f'FOM Hankel-norm: {so_sys.hankel_norm():e}')

    # Model order reduction
    run_mor_method(so_sys, w, SOBTpReductor(so_sys), 'SOBTp', r)
    run_mor_method(so_sys, w, SOBTvReductor(so_sys), 'SOBTv', r)
    run_mor_method(so_sys, w, SOBTpvReductor(so_sys), 'SOBTpv', r)
    run_mor_method(so_sys, w, SOBTvpReductor(so_sys), 'SOBTvp', r)
    run_mor_method(so_sys, w, SOBTfvReductor(so_sys), 'SOBTfv', r)
    run_mor_method(so_sys, w, SOBTReductor(so_sys), 'SOBT', r)
    run_mor_method(so_sys, w, SORIRKAReductor(so_sys), 'SOR-IRKA', r,
                   irka_options={'maxit': 10})
    run_mor_method(so_sys, w, BTReductor(so_sys.to_lti()), 'BT', r)
    run_mor_method(so_sys, w, IRKAReductor(so_sys.to_lti()), 'IRKA', r)


if __name__ == '__main__':
    run(main)

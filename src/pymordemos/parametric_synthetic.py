#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.core.config import config
from pymor.models.iosys import LTIModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor
from pymordemos.parametric_heat import run_mor_method_param


def main(
        n: int = Argument(100, help='Order of the FOM.'),
        r: int = Argument(10, help='Order of the ROMs.'),
):
    """Synthetic parametric demo.

    See the `MOR Wiki page <http://modelreduction.org/index.php/Synthetic_parametric_model>`_.
    """
    # Model
    # set coefficients
    a = -np.linspace(1e1, 1e3, n // 2)
    b = np.linspace(1e1, 1e3, n // 2)
    c = np.ones(n // 2)
    d = np.zeros(n // 2)

    # build 2x2 submatrices
    aa = np.empty(n)
    aa[::2] = a
    aa[1::2] = a
    bb = np.zeros(n)
    bb[::2] = b

    # set up system matrices
    Amu = sps.diags(aa, format='csc')
    A0 = sps.diags([bb, -bb], [1, -1], shape=(n, n), format='csc')
    B = np.zeros((n, 1))
    B[::2, 0] = 2
    C = np.empty((1, n))
    C[0, ::2] = c
    C[0, 1::2] = d

    # form operators
    A0 = NumpyMatrixOperator(A0)
    Amu = NumpyMatrixOperator(Amu)
    B = NumpyMatrixOperator(B)
    C = NumpyMatrixOperator(C)
    A = A0 + Amu * ProjectionParameterFunctional('mu')

    # form a model
    lti = LTIModel(A, B, C)

    mu_list = [1/50, 1/20, 1/10, 1/5, 1/2, 1]
    w = np.logspace(0.5, 3.5, 200)

    # System poles
    fig, ax = plt.subplots()
    for mu in mu_list:
        poles = lti.poles(mu=mu)
        ax.plot(poles.real, poles.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title('System poles')
    ax.legend()
    plt.show()

    # Magnitude plot
    fig, ax = plt.subplots()
    for mu in mu_list:
        lti.transfer_function.mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.legend()
    plt.show()

    # Hankel singular values
    fig, ax = plt.subplots()
    for mu in mu_list:
        hsv = lti.hsv(mu=mu)
        ax.semilogy(range(1, len(hsv) + 1), hsv, '.-', label=fr'$\mu = {mu}$')
    ax.set_title('Hankel singular values')
    ax.legend()
    plt.show()

    # System norms
    for mu in mu_list:
        print(f'mu = {mu}:')
        print(f'    H_2-norm of the full model:    {lti.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    H_inf-norm of the full model:  {lti.hinf_norm(mu=mu):e}')
        print(f'    Hankel-norm of the full model: {lti.hankel_norm(mu=mu):e}')

    # Model order reduction
    run_mor_method_param(lti, r, w, mu_list, BTReductor, 'BT')
    run_mor_method_param(lti, r, w, mu_list, IRKAReductor, 'IRKA')


if __name__ == "__main__":
    run(main)

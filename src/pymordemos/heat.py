#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

r"""1D heat equation demo

Discretization of the PDE:

.. math::
    :nowrap:

    \begin{align*}
        \partial_t z(x, t) &= \partial_{xx} z(x, t), & 0 < x < 1,\ t > 0 \\
        \partial_x z(0, t) &= z(0, t) - u(t), & t > 0 \\
        \partial_x z(1, t) &= -z(1, t), & t > 0 \\
        y(t) &= z(1, t), & t > 0
    \end{align*}

where :math:`u(t)` is the input and :math:`y(t)` is the output.
"""

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from pymor.basic import *
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor

import logging
logging.getLogger('pymor.algorithms.gram_schmidt.gram_schmidt').setLevel(logging.ERROR)

if __name__ == '__main__':
    # dimension of the system
    n = 100

    p = InstationaryProblem(
        StationaryProblem(
            domain=LineDomain([0.,1.], left='robin', right='robin'),
            diffusion=ConstantFunction(1., 1),
            robin_data=(ConstantFunction(1., 1), ExpressionFunction('(x[...,0] < 0.5) * 1.', 1)),
            functionals={'output': ('l2_boundary', ExpressionFunction('(x[...,0] > 0.5) * 1.', 1))}
        ),
        ConstantFunction(0., 1),
        T=3.
    )

    d, _ = discretize_instationary_cg(p, diameter=1/(n-1), nt=100)

    U = d.solve()
    print(U[-1].to_numpy().ravel())
    d.visualize(d.solve())

    lti = d.to_lti()

    print('n = {}'.format(lti.n))
    print('m = {}'.format(lti.m))
    print('p = {}'.format(lti.p))

    # System poles
    poles = lti.poles()
    fig, ax = plt.subplots()
    ax.plot(poles.real, poles.imag, '.')
    ax.set_title('System poles')
    plt.show()

    # Bode plot of the full model
    w = np.logspace(-2, 3, 100)
    fig, ax = plt.subplots()
    lti.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the full model')
    plt.show()

    # Hankel singular values
    hsv = lti.hsv()
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
    ax.set_title('Hankel singular values')
    plt.show()

    # Norms of the system
    print('H_2-norm of the full model:    {:e}'.format(lti.h2_norm()))
    print('H_inf-norm of the full model:  {:e}'.format(lti.hinf_norm()))
    print('Hankel-norm of the full model: {:e}'.format(lti.hankel_norm()))

    # Balanced Truncation
    r = 5
    reductor = BTReductor(lti)
    rom_bt = reductor.reduce(r, tol=1e-5)
    err_bt = lti - rom_bt
    print('H_2-error for the BT ROM:    {:e}'.format(err_bt.h2_norm()))
    print('H_inf-error for the BT ROM:  {:e}'.format(err_bt.hinf_norm()))
    print('Hankel-error for the BT ROM: {:e}'.format(err_bt.hankel_norm()))

    # Bode plot of the full and BT reduced model
    fig, ax = plt.subplots()
    lti.mag_plot(w, ax=ax)
    rom_bt.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and BT reduced model')
    plt.show()

    # Bode plot of the BT error system
    fig, ax = plt.subplots()
    err_bt.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the BT error system')
    plt.show()

    # Iterative Rational Krylov Algorithm
    sigma = np.logspace(-1, 3, r)
    tol = 1e-4
    maxit = 100
    irka_reductor = IRKAReductor(lti)
    rom_irka = irka_reductor.reduce(r, sigma, tol=tol, maxit=maxit, compute_errors=True)

    # Shift distances
    fig, ax = plt.subplots()
    ax.semilogy(irka_reductor.dist, '.-')
    ax.set_title('Distances between shifts in IRKA iterations')
    plt.show()

    err_irka = lti - rom_irka
    print('H_2-error for the IRKA ROM:    {:e}'.format(err_irka.h2_norm()))
    print('H_inf-error for the IRKA ROM:  {:e}'.format(err_irka.hinf_norm()))
    print('Hankel-error for the IRKA ROM: {:e}'.format(err_irka.hankel_norm()))

    # Bode plot of the full and IRKA reduced model
    fig, ax = plt.subplots()
    lti.mag_plot(w, ax=ax)
    rom_irka.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and IRKA reduced model')
    plt.show()

    # Bode plot of the IRKA error system
    fig, ax = plt.subplots()
    err_irka.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the IRKA error system')
    plt.show()

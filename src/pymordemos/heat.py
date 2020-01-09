#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

r"""2D heat equation demo

Discretization of the PDE:

.. math::
    :nowrap:

    \begin{align*}
        \partial_t z(x, y, t) &= \Delta z(x, y, t),      & 0 < x, y < 1,\ t > 0 \\
        -\nabla z(0, y, t) \cdot n &= z(0, y, t) - u(t), & 0 < y < 1, t > 0 \\
        -\nabla z(1, y, t) \cdot n &= z(1, y, t),        & 0 < y < 1, t > 0 \\
        -\nabla z(0, x, t) \cdot n &= z(0, x, t),        & 0 < x < 1, t > 0 \\
        -\nabla z(1, x, t) \cdot n &= z(1, x, t),        & 0 < x < 1, t > 0 \\
        z(x, y, 0) &= 0                                  & 0 < x, y < 1 \\
        y(t) &= \int_0^1 z(1, y, t) dy,                  & t > 0 \\
    \end{align*}

where :math:`u(t)` is the input and :math:`y(t)` is the output.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymor.basic import (InstationaryProblem, StationaryProblem, RectDomain, ConstantFunction, ExpressionFunction,
                         discretize_instationary_cg, BTReductor, IRKAReductor)
from pymor.core.config import config

import logging
logging.getLogger('pymor.algorithms.gram_schmidt.gram_schmidt').setLevel(logging.ERROR)


if __name__ == '__main__':
    p = InstationaryProblem(
        StationaryProblem(
            domain=RectDomain([[0., 0.], [1., 1.]], left='robin', right='robin', top='robin', bottom='robin'),
            diffusion=ConstantFunction(1., 2),
            robin_data=(ConstantFunction(1., 2), ExpressionFunction('(x[...,0] < 1e-10) * 1.', 2)),
            outputs=[('l2_boundary', ExpressionFunction('(x[...,0] > (1 - 1e-10)) * 1.', 2))]
        ),
        ConstantFunction(0., 2),
        T=1.
    )

    fom, _ = discretize_instationary_cg(p, diameter=1/10, nt=100)

    fom.visualize(fom.solve())

    lti = fom.to_lti()

    print(f'order of the model = {lti.order}')
    print(f'number of inputs   = {lti.input_dim}')
    print(f'number of outputs  = {lti.output_dim}')

    # System poles
    poles = lti.poles()
    fig, ax = plt.subplots()
    ax.plot(poles.real, poles.imag, '.')
    ax.set_title('System poles')
    plt.show()

    # Magnitude plot of the full model
    w = np.logspace(-1, 3, 100)
    fig, ax = plt.subplots()
    lti.mag_plot(w, ax=ax)
    ax.set_title('Magnitude plot of the full model')
    plt.show()

    # Hankel singular values
    hsv = lti.hsv()
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
    ax.set_title('Hankel singular values')
    plt.show()

    # Norms of the system
    print(f'FOM H_2-norm:    {lti.h2_norm():e}')
    if config.HAVE_SLYCOT:
        print(f'FOM H_inf-norm:  {lti.hinf_norm():e}')
    else:
        print('Skipped H_inf-norm calculation due to missing slycot.')
    print(f'FOM Hankel-norm: {lti.hankel_norm():e}')

    # Balanced Truncation
    r = 5
    reductor = BTReductor(lti)
    rom_bt = reductor.reduce(r, tol=1e-5)
    err_bt = lti - rom_bt
    print(f'BT relative H_2-error:    {err_bt.h2_norm() / lti.h2_norm():e}')
    if config.HAVE_SLYCOT:
        print(f'BT relative H_inf-error:  {err_bt.hinf_norm() / lti.hinf_norm():e}')
    else:
        print('Skipped H_inf-norm calculation due to missing slycot.')
    print(f'BT relative Hankel-error: {err_bt.hankel_norm() / lti.hankel_norm():e}')

    # Magnitude plot of the full and BT reduced model
    fig, ax = plt.subplots()
    lti.mag_plot(w, ax=ax)
    rom_bt.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Magnitude plot of the full and BT reduced model')
    plt.show()

    # Magnitude plot of the BT error system
    fig, ax = plt.subplots()
    err_bt.mag_plot(w, ax=ax)
    ax.set_title('Magnitude plot of the BT error system')
    plt.show()

    # Iterative Rational Krylov Algorithm
    irka_reductor = IRKAReductor(lti)
    rom_irka = irka_reductor.reduce(r, compute_errors=True)

    # Shift distances
    fig, ax = plt.subplots()
    ax.semilogy(irka_reductor.conv_crit, '.-')
    ax.set_title('Distances between shifts in IRKA iterations')
    plt.show()

    err_irka = lti - rom_irka
    print(f'IRKA relative H_2-error:    {err_irka.h2_norm() / lti.h2_norm():e}')
    if config.HAVE_SLYCOT:
        print(f'IRKA relative H_inf-error:  {err_irka.hinf_norm() / lti.hinf_norm():e}')
    else:
        print('Skipped H_inf-norm calculation due to missing slycot.')
    print(f'IRKA relative Hankel-error: {err_irka.hankel_norm() / lti.hankel_norm():e}')

    # Magnitude plot of the full and IRKA reduced model
    fig, ax = plt.subplots()
    lti.mag_plot(w, ax=ax)
    rom_irka.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Magnitude plot of the full and IRKA reduced model')
    plt.show()

    # Magnitude plot of the IRKA error system
    fig, ax = plt.subplots()
    err_irka.mag_plot(w, ax=ax)
    ax.set_title('Magnitude plot of the IRKA error system')
    plt.show()

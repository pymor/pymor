#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.core.config import config
from pymor.core.logger import set_log_levels
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.reductors.bt import BTReductor, LQGBTReductor, BRBTReductor
from pymor.reductors.h2 import IRKAReductor, TSIAReductor, OneSidedIRKAReductor


def run_mor_method(lti, w, reductor, reductor_short_name, r, **reduce_kwargs):
    """Run a model order reduction method.

    Parameters
    ----------
    lti
        The full-order |LTIModel|.
    w
        Array of frequencies.
    reductor
        The reductor object.
    reductor_short_name
        A short name for the reductor.
    r
        The order of the reduced-order model.
    reduce_kwargs
        Optional keyword arguments for the reduce method.
    """
    # Reduction
    rom = reductor.reduce(r, **reduce_kwargs)
    err = lti - rom

    # Poles of the reduced-order model
    poles_rom = rom.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom.real, poles_rom.imag, '.')
    ax.set_title(f"{reductor_short_name} reduced model's poles")
    plt.show()

    # Errors
    print(f'{reductor_short_name} relative H_2-error:    {err.h2_norm() / lti.h2_norm():e}')
    if config.HAVE_SLYCOT:
        print(f'{reductor_short_name} relative H_inf-error:  {err.hinf_norm() / lti.hinf_norm():e}')
    else:
        print('Skipped H_inf-norm calculation due to missing slycot.')
    print(f'{reductor_short_name} relative Hankel-error: {err.hankel_norm() / lti.hankel_norm():e}')

    # Magnitude plot of the full and reduced model
    fig, ax = plt.subplots()
    lti.mag_plot(w, ax=ax)
    rom.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title(f'Magnitude plot of the full and {reductor_short_name} reduced model')
    plt.show()

    # Magnitude plot of the error system
    fig, ax = plt.subplots()
    err.mag_plot(w, ax=ax)
    ax.set_title(f'Magnitude plot of the {reductor_short_name} error system')
    plt.show()


def main(
        diameter: float = Argument(0.1, help='Diameter option for the domain discretizer.'),
        r: int = Argument(5, help='Order of the ROMs.'),
):
    r"""2D heat equation demo.

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
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

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

    fom, _ = discretize_instationary_cg(p, diameter=diameter, nt=100)

    fom.visualize(fom.solve())

    lti = fom.to_lti()

    print(f'order of the model = {lti.order}')
    print(f'number of inputs   = {lti.dim_input}')
    print(f'number of outputs  = {lti.dim_output}')

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

    # Model order reduction
    run_mor_method(lti, w, BTReductor(lti), 'BT', r, tol=1e-5)
    run_mor_method(lti, w, LQGBTReductor(lti), 'LQGBT', r, tol=1e-5)
    run_mor_method(lti, w, BRBTReductor(lti), 'BRBT', r, tol=1e-5)
    run_mor_method(lti, w, IRKAReductor(lti), 'IRKA', r)
    run_mor_method(lti, w, TSIAReductor(lti), 'TSIA', r)
    run_mor_method(lti, w, OneSidedIRKAReductor(lti, 'V'), 'OS-IRKA', r)


if __name__ == '__main__':
    run(main)

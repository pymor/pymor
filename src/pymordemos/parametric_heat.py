#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.analyticalproblems.domaindescriptions import LineDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.core.config import config
from pymor.core.logger import set_log_levels
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.bt import BTReductor, LQGBTReductor, BRBTReductor
from pymor.reductors.h2 import IRKAReductor, TSIAReductor, OneSidedIRKAReductor
from pymor.reductors.mt import MTReductor


def run_mor_method_param(fom, r, w, mus, reductor_cls, reductor_short_name, **reductor_kwargs):
    """Plot reductor errors for different parameter values.

    Parameters
    ----------
    fom
        The full-order |LTIModel|.
    r
        The order of the reduced-order model.
    w
        Array of frequencies.
    mus
        An array of parameter values.
    reductor_cls
        The reductor class.
    reductor_short_name
        A short name for the reductor.
    reductor_kwargs
        Optional keyword arguments for the reductor class.
    """
    # Reduction
    roms = []
    for mu in mus:
        rom = reductor_cls(fom, mu=mu, **reductor_kwargs).reduce(r)
        roms.append(rom)

    # Poles
    fig, ax = plt.subplots()
    for rom in roms:
        poles_rom = rom.poles()
        ax.plot(poles_rom.real, poles_rom.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title(f"{reductor_short_name} reduced model's poles")
    plt.show()

    # Magnitude plots
    fig, ax = plt.subplots()
    for mu, rom in zip(mus, roms):
        rom.transfer_function.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title(f'Magnitude plot of {reductor_short_name} reduced models')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    for mu, rom in zip(mus, roms):
        (fom - rom).transfer_function.mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title(f'Magnitude plot of the {reductor_short_name} error system')
    ax.legend()
    plt.show()

    # Errors
    for mu, rom in zip(mus, roms):
        err = fom - rom
        print(f'mu = {mu}')
        print(f'    {reductor_short_name} relative H_2-error:'
              f'    {err.h2_norm(mu=mu) / fom.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    {reductor_short_name} relative H_inf-error:'
                  f'  {err.hinf_norm(mu=mu) / fom.hinf_norm(mu=mu):e}')
        print(f'    {reductor_short_name} relative Hankel-error:'
              f' {err.hankel_norm(mu=mu) / fom.hankel_norm(mu=mu):e}')


def main(
        diameter: float = Argument(0.01, help='Diameter option for the domain discretizer.'),
        r: int = Argument(5, help='Order of the ROMs.'),
):
    """Parametric 1D heat equation example."""
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})
    # This demo opens more than 20 figures and matplotlib wants to warn us about that
    plt.rcParams['figure.max_open_warning'] = 0

    # Model
    p = InstationaryProblem(
        StationaryProblem(
            domain=LineDomain([0., 1.], left='robin', right='robin'),
            diffusion=LincombFunction([ExpressionFunction('(x[0] <= 0.5) * 1.', 1),
                                       ExpressionFunction('(0.5 < x[0]) * 1.', 1)],
                                      [1,
                                       ProjectionParameterFunctional('diffusion')]),
            robin_data=(ConstantFunction(1., 1), ExpressionFunction('(x[0] < 1e-10) * 1.', 1)),
            outputs=(('l2_boundary', ExpressionFunction('(x[0] > (1 - 1e-10)) * 1.', 1)),),
        ),
        ConstantFunction(0., 1),
        T=3.
    )

    fom, _ = discretize_instationary_cg(p, diameter=diameter, nt=100)

    fom.visualize(fom.solve(mu=0.1))
    fom.visualize(fom.solve(mu=1))
    fom.visualize(fom.solve(mu=10))

    lti = fom.to_lti()

    print(f'order of the model = {lti.order}')
    print(f'number of inputs   = {lti.dim_input}')
    print(f'number of outputs  = {lti.dim_output}')

    mu_list = [0.1, 1, 10]
    w_list = np.logspace(-1, 3, 100)

    # System poles
    fig, ax = plt.subplots()
    for mu in mu_list:
        poles = lti.poles(mu=mu)
        ax.plot(poles.real, poles.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title('System poles')
    ax.legend()
    plt.show()

    # Magnitude plots
    fig, ax = plt.subplots()
    for mu in mu_list:
        lti.transfer_function.mag_plot(w_list, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the full model')
    ax.legend()
    plt.show()

    # Hankel singular values
    fig, ax = plt.subplots()
    for mu in mu_list:
        hsv = lti.hsv(mu=mu)
        ax.semilogy(range(1, len(hsv) + 1), hsv, label=fr'$\mu = {mu}$')
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
    run_mor_method_param(lti, r, w_list, mu_list, BTReductor, 'BT')
    run_mor_method_param(lti, r, w_list, mu_list, LQGBTReductor, 'LQGBT')
    run_mor_method_param(lti, r, w_list, mu_list, BRBTReductor, 'BRBT')
    run_mor_method_param(lti, r, w_list, mu_list, IRKAReductor, 'IRKA')
    run_mor_method_param(lti, r, w_list, mu_list, TSIAReductor, 'TSIA')
    run_mor_method_param(lti, r, w_list, mu_list, OneSidedIRKAReductor, 'OS-IRKA', version='V')
    run_mor_method_param(lti, r, w_list, mu_list, MTReductor, 'MT')


if __name__ == "__main__":
    run(main)

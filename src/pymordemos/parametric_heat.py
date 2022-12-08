#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.analyticalproblems.domaindescriptions import LineDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.core.config import config
from pymor.core.logger import set_log_levels
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.models.iosys import LTIModel, SecondOrderModel
from pymor.models.transfer_function import TransferFunction
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.bt import BTReductor, LQGBTReductor, BRBTReductor
from pymor.reductors.h2 import IRKAReductor, TSIAReductor, OneSidedIRKAReductor
from pymor.reductors.mt import MTReductor


def fom_properties_param(fom, w, mus):
    """Show properties of the full-order model.

    Parameters
    ----------
    fom
        The full-order `Model` from :mod:`~pymor.models.iosys` or a |TransferFunction|.
    w
        Array of frequencies.
    mus
        List of parameter values.
    """
    # Model info
    print(fom)

    # System norms
    if not isinstance(fom, TransferFunction):
        for mu in mus:
            print(f'mu = {mu}:')
            print(f'    H_2-norm of the full model:    {fom.h2_norm(mu=mu):e}')
            if config.HAVE_SLYCOT:
                print(f'    H_inf-norm of the full model:  {fom.hinf_norm(mu=mu):e}')
            print(f'    Hankel-norm of the full model: {fom.hankel_norm(mu=mu):e}')

    # Figure
    if isinstance(fom, (LTIModel, SecondOrderModel)):
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        fig.suptitle('Full-order model')
        subfigs = fig.subfigures(1, 2)
        subfigs1 = subfigs[1].subfigures(2, 1)
        fig_bode = subfigs[0]
        fig_poles = subfigs1[0]
        fig_sv = subfigs1[1]
    else:  # TransferFunction
        fig_bode = plt.figure(figsize=(5, 8), constrained_layout=True)
    markers = 'ox+1234'

    # Bode plots
    axs = fig_bode.subplots(2, 1, squeeze=False)
    for mu in mus:
        if isinstance(fom, TransferFunction):
            fom.bode_plot(w, ax=axs, mu=mu, label=fr'$\mu = {mu}$')
        else:
            fom.transfer_function.bode_plot(w, ax=axs, mu=mu, label=fr'$\mu = {mu}$')
    for ax in axs.flat:
        ax.legend()

    # System poles
    if not isinstance(fom, TransferFunction):
        ax = fig_poles.subplots()
        for mu, marker in zip(mus, markers):
            poles = fom.poles(mu=mu)
            ax.plot(poles.real, poles.imag, marker, fillstyle='none', label=fr'$\mu = {mu}$')
        ax.set_title('System poles')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imag')
        ax.legend()

    # Hankel singular values
    if isinstance(fom, LTIModel):
        ax = fig_sv.subplots()
        for mu, marker in zip(mus, markers):
            hsv = fom.hsv(mu=mu)
            ax.semilogy(range(1, len(hsv) + 1), hsv, f'{marker}-', fillstyle='none', label=fr'$\mu = {mu}$')
        ax.set_title('Hankel singular values')
        ax.set_xlabel('Index')
        ax.legend()
    elif isinstance(fom, SecondOrderModel):
        axs = fig_sv.subplots(2, 2, sharex=True, sharey=True)
        for mu in mus:
            psv = fom.psv(mu=mu)
            vsv = fom.vsv(mu=mu)
            pvsv = fom.pvsv(mu=mu)
            vpsv = fom.vpsv(mu=mu)
            axs[0, 0].semilogy(range(1, len(psv) + 1), psv, '.-')
            axs[0, 1].semilogy(range(1, len(vsv) + 1), vsv, '.-', label=fr'$\mu = {mu}$')
            axs[1, 0].semilogy(range(1, len(pvsv) + 1), pvsv, '.-')
            axs[1, 1].semilogy(range(1, len(vpsv) + 1), vpsv, '.-')
        axs[0, 0].set_title('Position s.v.')
        axs[0, 1].set_title('Velocity s.v.')
        axs[1, 0].set_title('Position-velocity s.v.')
        axs[1, 0].set_xlabel('Index')
        axs[1, 1].set_title('Velocity-position s.v.')
        axs[1, 1].set_xlabel('Index')
        axs[0, 1].legend()
    plt.show()


def run_mor_method_param(fom, r, w, mus, reductor_cls, reductor_short_name, **reductor_kwargs):
    """Plot reductor errors for different parameter values.

    Parameters
    ----------
    fom
        The full-order `Model` from :mod:`~pymor.models.iosys` or a |TransferFunction|.
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

    # Errors
    if not isinstance(fom, TransferFunction):
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

    # Figure and subfigures
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f'{reductor_short_name} reduced-order model')
    subfigs = fig.subfigures(1, 2)
    subfigs1 = subfigs[1].subfigures(2, 1)
    fig_bode = subfigs[0]
    fig_poles = subfigs1[0]
    fig_mag = subfigs1[1]

    # Bode plots of reduced-order models
    axs = fig_bode.subplots(2 * fom.dim_output, fom.dim_input, squeeze=False)
    for mu, rom in zip(mus, roms):
        rom.transfer_function.bode_plot(w, ax=axs, label=fr'$\mu = {mu}$')
    for ax in axs.flat:
        ax.legend()

    # Poles
    ax = fig_poles.subplots()
    for mu, rom, marker in zip(mus, roms, 'ox+1234'):
        poles_rom = rom.poles()
        ax.plot(poles_rom.real, poles_rom.imag, marker, fillstyle='none', label=fr'$\mu = {mu}$')
    ax.set_title("ROM's poles")
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.legend()

    # Magnitude plot of the error systems
    ax = fig_mag.subplots()
    for mu, rom in zip(mus, roms):
        if isinstance(fom, TransferFunction):
            (fom - rom).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
        else:
            (fom - rom).transfer_function.mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the error system')
    ax.legend()
    plt.show()


def main(
        diameter: float = Argument(0.01, help='Diameter option for the domain discretizer.'),
        r: int = Argument(5, help='Order of the ROMs.'),
):
    """Parametric 1D heat equation example."""
    set_log_levels({
        'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING',
        'pymor.algorithms.lradi.solve_lyap_lrcf': 'WARNING',
        'pymor.reductors.basic.LTIPGReductor': 'WARNING',
    })
    plt.rcParams['axes.grid'] = True

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

    mus = [0.1, 1, 10]
    w = (1e-1, 1e3)
    fom_properties_param(lti, w, mus)

    # Model order reduction
    run_mor_method_param(lti, r, w, mus, BTReductor, 'BT')
    run_mor_method_param(lti, r, w, mus, LQGBTReductor, 'LQGBT')
    run_mor_method_param(lti, r, w, mus, BRBTReductor, 'BRBT')
    run_mor_method_param(lti, r, w, mus, IRKAReductor, 'IRKA')
    run_mor_method_param(lti, r, w, mus, TSIAReductor, 'TSIA')
    run_mor_method_param(lti, r, w, mus, OneSidedIRKAReductor, 'OS-IRKA', version='V')
    run_mor_method_param(lti, r, w, mus, MTReductor, 'MT')


if __name__ == "__main__":
    run(main)

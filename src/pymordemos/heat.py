#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import matplotlib.pyplot as plt
import numpy as np
from typer import Argument, run

from pymor.algorithms.lradi import lyap_lrcf_solver_options
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.core.config import config
from pymor.core.logger import set_log_levels
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.models.iosys import LTIModel, SecondOrderModel
from pymor.reductors.bt import BRBTReductor, BTReductor, LQGBTReductor
from pymor.reductors.h2 import IRKAReductor, OneSidedIRKAReductor, TSIAReductor
from pymor.reductors.mt import MTReductor


def fom_properties(fom, w, stable_lti=True):
    """Show properties of the full-order model.

    Parameters
    ----------
    fom
        The full-order `Model` from :mod:`~pymor.models.iosys` or a |TransferFunction|.
    w
        Array of frequencies.
    stable_lti
        Whether the FOM is stable (assuming it is an |LTIModel|).
    """
    from pymor.models.transfer_function import TransferFunction
    if not isinstance(fom, TransferFunction):
        print(f'order of the model = {fom.order}')
    print(f'number of inputs   = {fom.dim_input}')
    print(f'number of outputs  = {fom.dim_output}')

    # System norms
    if stable_lti:
        print(f'FOM H_2-norm:    {fom.h2_norm():e}')
        if not isinstance(fom, TransferFunction):
            if config.HAVE_SLYCOT:
                print(f'FOM H_inf-norm:  {fom.hinf_norm():e}')
            else:
                print('Skipped H_inf-norm calculation due to missing slycot.')
            print(f'FOM Hankel-norm: {fom.hankel_norm():e}')
    else:
        print(f'FOM L_2-norm:    {fom.l2_norm():e}')
        if config.HAVE_SLYCOT:
            print(f'FOM L_inf-norm:  {fom.linf_norm():e}')
        else:
            print('Skipped L_inf-norm calculation due to missing slycot.')

    # Figure
    if isinstance(fom, LTIModel) and fom.T is not None:
        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        fig.suptitle('Full-order model')
        subfigs = fig.subfigures(1, 3)
        subfigs1 = subfigs[1].subfigures(2, 1)
        fig_bode = subfigs[0]
        fig_poles = subfigs1[0]
        fig_sv = subfigs1[1]
        fig_time = subfigs[2]
    elif isinstance(fom, (LTIModel, SecondOrderModel)):
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        fig.suptitle('Full-order model')
        subfigs = fig.subfigures(1, 2)
        subfigs1 = subfigs[1].subfigures(2, 1)
        fig_bode = subfigs[0]
        fig_poles = subfigs1[0]
        fig_sv = subfigs1[1]
        fig_time = None
    else:  # TransferFunction
        fig_bode = plt.figure(figsize=(5, 8), constrained_layout=True)
        fig_poles = None
        fig_sv = None
        fig_time = None

    # Bode plot
    if fig_bode is not None:
        ax = fig_bode.subplots(2 * fom.dim_output, fom.dim_input, squeeze=False)
        if isinstance(fom, TransferFunction):
            fom.bode_plot(w, ax=ax)
        else:
            fom.transfer_function.bode_plot(w, ax=ax)

    # System poles
    if fig_poles is not None:
        poles = fom.poles()
        ax = fig_poles.subplots()
        ax.plot(poles.real, poles.imag, '.')
        ax.set_title('System poles')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imag')

        if not stable_lti:
            ast_spectrum = fom.get_ast_spectrum()
            print(f'Anti-stable system poles:  {ast_spectrum[1]}')

    # Hankel singular values
    if fig_sv is not None:
        if isinstance(fom, LTIModel):
            if stable_lti:
                hsv = fom.hsv()
            else:
                hsv = fom._sv_U_V(typ='bs')[0]
            ax = fig_sv.subplots()
            ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
            if stable_lti:
                ax.set_title('Hankel singular values')
            else:
                ax.set_title('Bernoulli stabilized singular values')
            ax.set_xlabel('Index')
        else:  # SecondOrderModel
            psv = fom.psv()
            vsv = fom.vsv()
            pvsv = fom.pvsv()
            vpsv = fom.vpsv()
            axs = fig_sv.subplots(2, 2, sharex=True, sharey=True)
            axs[0, 0].semilogy(range(1, len(psv) + 1), psv, '.-')
            axs[0, 0].set_title('Position s.v.')
            axs[0, 1].semilogy(range(1, len(vsv) + 1), vsv, '.-')
            axs[0, 1].set_title('Velocity s.v.')
            axs[1, 0].semilogy(range(1, len(pvsv) + 1), pvsv, '.-')
            axs[1, 0].set_title('Position-velocity s.v.')
            axs[1, 0].set_xlabel('Index')
            axs[1, 1].semilogy(range(1, len(vpsv) + 1), vpsv, '.-')
            axs[1, 1].set_title('Velocity-position s.v.')
            axs[1, 1].set_xlabel('Index')

    # Time response
    if fig_time is not None:
        fig_i, fig_s = fig_time.subfigures(2, 1)
        fig_i.suptitle('Impulse response')
        fig_s.suptitle('Step response')
        axs_i = fig_i.subplots(fom.dim_output, fom.dim_input, sharex=True, sharey=True, squeeze=False)
        axs_s = fig_s.subplots(fom.dim_output, fom.dim_input, sharex=True, sharey=True, squeeze=False)
        y_i = fom.impulse_resp()
        y_s = fom.step_resp()
        times = np.linspace(0, fom.T, fom.time_stepper.nt + 1)
        for i in range(fom.dim_output):
            for j in range(fom.dim_input):
                axs_i[i, j].plot(times, y_i[:, i, j])
                axs_s[i, j].plot(times, y_s[:, i, j])
        for j in range(fom.dim_input):
            axs_i[-1, j].set_xlabel('Time (s)')
            axs_s[-1, j].set_xlabel('Time (s)')

    plt.show()


def run_mor_method(fom, w, reductor, reductor_short_name, r, stable=True, **reduce_kwargs):
    """Run a model order reduction method.

    Parameters
    ----------
    fom
        The full-order `Model` from :mod:`~pymor.models.iosys` or a |TransferFunction|.
    w
        Array of frequencies.
    reductor
        The reductor object.
    reductor_short_name
        A short name for the reductor.
    r
        The order of the reduced-order model.
    stable
        Whether the FOM is stable.
    reduce_kwargs
        Optional keyword arguments for the reduce method.
    """
    # Reduction
    rom = reductor.reduce(r, **reduce_kwargs)
    err = fom - rom
    if isinstance(err, LTIModel):
        solver_options = {'lyap_lrcf': lyap_lrcf_solver_options(lradi_shifts='projection_shifts')['lradi']}
        err = err.with_(solver_options=solver_options)

    # Errors
    from pymor.models.transfer_function import TransferFunction
    if not isinstance(fom, TransferFunction):
        if stable:
            print(f'{reductor_short_name} relative H_2-error:    {err.h2_norm() / fom.h2_norm():e}')
            if config.HAVE_SLYCOT:
                print(f'{reductor_short_name} relative H_inf-error:  {err.hinf_norm() / fom.hinf_norm():e}')
            else:
                print('Skipped H_inf-norm calculation due to missing slycot.')
            print(f'{reductor_short_name} relative Hankel-error: {err.hankel_norm() / fom.hankel_norm():e}')
        else:
            if config.HAVE_SLYCOT:
                print(f'{reductor_short_name} relative L_inf-error:  {err.linf_norm() / fom.linf_norm():e}')
            else:
                print('Skipped L_inf-norm calculation due to missing slycot.')
    elif isinstance(rom, LTIModel):
        error = np.sqrt(fom.h2_norm()**2 - 2 * fom.h2_inner(rom).real + rom.h2_norm()**2)
        print(f'{reductor_short_name} relative H_2-error:    {error / fom.h2_norm():e}')
    else:
        print(f'{reductor_short_name} relative H_2-error:    {err.h2_norm() / fom.h2_norm():e}')

    # Figure and subfigures
    if isinstance(fom, LTIModel) and fom.T is not None:
        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        subfigs = fig.subfigures(1, 3)
        subfigs1 = subfigs[1].subfigures(2, 1)
        fig_bode = subfigs[0]
        fig_poles = subfigs1[0]
        fig_err = subfigs1[1]
        fig_time = subfigs[2]
    else:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        subfigs = fig.subfigures(1, 2)
        subfigs1 = subfigs[1].subfigures(2, 1)
        fig_bode = subfigs[0]
        fig_poles = subfigs1[0]
        fig_err = subfigs1[1]
        fig_time = None
    fig.suptitle(f'{reductor_short_name} reduced-order model')

    # Bode plot of the full and reduced model
    axs = fig_bode.subplots(2 * fom.dim_output, fom.dim_input, squeeze=False)
    if isinstance(fom, TransferFunction):
        fom.bode_plot(w, ax=axs, label='FOM')
    else:
        fom.transfer_function.bode_plot(w, ax=axs, label='FOM')
    rom.transfer_function.bode_plot(w, ax=axs, linestyle='dashed', label='ROM')
    for ax in axs.flat:
        ax.legend()

    # Poles of the reduced-order model
    poles_rom = rom.poles()
    ax = fig_poles.subplots()
    ax.plot(poles_rom.real, poles_rom.imag, '.')
    ax.set_title("ROM's poles")
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')

    # Magnitude plot of the error system
    ax = fig_err.subplots()
    if isinstance(err, TransferFunction):
        err.mag_plot(w, ax=ax)
    else:
        err.transfer_function.mag_plot(w, ax=ax)
    ax.set_title('Magnitude plot of the error system')

    # Time response of the (full and) reduced model
    if fig_time is not None:
        fig_i, fig_s = fig_time.subfigures(2, 1)
        fig_i.suptitle('Impulse response')
        fig_s.suptitle('Step response')
        axs_i = fig_i.subplots(fom.dim_output, fom.dim_input, sharex=True, sharey=True, squeeze=False)
        axs_s = fig_s.subplots(fom.dim_output, fom.dim_input, sharex=True, sharey=True, squeeze=False)
        y_i = fom.impulse_resp()
        y_s = fom.step_resp()
        yr_i = rom.impulse_resp()
        yr_s = rom.step_resp()
        times = np.linspace(0, fom.T, fom.time_stepper.nt + 1)
        for i in range(fom.dim_output):
            for j in range(fom.dim_input):
                axs_i[i, j].plot(times, y_i[:, i, j], label='FOM')
                axs_s[i, j].plot(times, y_s[:, i, j], label='FOM')
                axs_i[i, j].plot(times, yr_i[:, i, j], '--', label='ROM')
                axs_s[i, j].plot(times, yr_s[:, i, j], '--', label='ROM')
        for j in range(fom.dim_input):
            axs_i[-1, j].set_xlabel('Time (s)')
            axs_s[-1, j].set_xlabel('Time (s)')
        for ax in axs_i.flat:
            ax.legend()
        for ax in axs_s.flat:
            ax.legend()

    plt.show()


def main(
        diameter: float = Argument(0.1, help='Diameter option for the domain discretizer.'),
        r: int = Argument(5, help='Order of the ROMs.'),
):
    r"""2D heat equation demo.

    Discretization of the PDE:

    .. math::
        \begin{align*}
            \partial_t z(x, y, t) &= \Delta z(x, y, t),      & 0 < x, y < 1,\ t > 0 \\
            -\nabla z(0, y, t) \cdot n &= z(0, y, t) - u(t), & 0 < y < 1, t > 0 \\
            -\nabla z(1, y, t) \cdot n &= z(1, y, t),        & 0 < y < 1, t > 0 \\
            -\nabla z(x, 0, t) \cdot n &= z(x, 0, t),        & 0 < x < 1, t > 0 \\
            -\nabla z(x, 1, t) \cdot n &= z(x, 1, t),        & 0 < x < 1, t > 0 \\
            z(x, y, 0) &= 0                                  & 0 < x, y < 1 \\
            y(t) &= \int_0^1 z(1, y, t) dy,                  & t > 0
        \end{align*}

    where :math:`u(t)` is the input and :math:`y(t)` is the output.
    """
    set_log_levels({
        'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING',
        'pymor.algorithms.lradi.solve_lyap_lrcf': 'WARNING',
        'pymor.reductors.basic.LTIPGReductor': 'WARNING',
    })
    plt.rcParams['axes.grid'] = True

    p = InstationaryProblem(
        StationaryProblem(
            domain=RectDomain([[0., 0.], [1., 1.]], left='robin', right='robin', top='robin', bottom='robin'),
            diffusion=ConstantFunction(1., 2),
            robin_data=(ConstantFunction(1., 2), ExpressionFunction('(x[0] < 1e-10) * 1.', 2)),
            outputs=[('l2_boundary', ExpressionFunction('(x[0] > (1 - 1e-10)) * 1.', 2))]
        ),
        ConstantFunction(0., 2),
        T=1.
    )

    fom, _ = discretize_instationary_cg(p, diameter=diameter, nt=100)

    fom.visualize(fom.solve())

    # LTI system
    solver_options = {'lyap_lrcf': lyap_lrcf_solver_options(lradi_shifts='wachspress_shifts')['lradi']}
    lti = fom.to_lti().with_(solver_options=solver_options, T=1, time_stepper=ImplicitEulerTimeStepper(100))

    # System properties
    w = (1e-1, 1e3)
    fom_properties(lti, w)

    # Model order reduction
    run_mor_method(lti, w, BTReductor(lti), 'BT', r, tol=1e-5)
    run_mor_method(lti, w, LQGBTReductor(lti), 'LQGBT', r, tol=1e-5)
    run_mor_method(lti, w, BRBTReductor(lti, gamma=0.2), 'BRBT', r, tol=1e-5)
    run_mor_method(lti, w, IRKAReductor(lti), 'IRKA', r)
    run_mor_method(lti, w, IRKAReductor(lti), 'IRKA (with Arnoldi)', r, projection='arnoldi')
    run_mor_method(lti, w, TSIAReductor(lti), 'TSIA', r)
    run_mor_method(lti, w, OneSidedIRKAReductor(lti, 'V'), 'OS-IRKA', r)
    run_mor_method(lti, w, MTReductor(lti), 'MT', r)


if __name__ == '__main__':
    run(main)

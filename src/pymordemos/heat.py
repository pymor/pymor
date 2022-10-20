#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt
from typer import Argument, run


from pymor.algorithms.lradi import lyap_lrcf_solver_options
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.core.config import config
from pymor.core.logger import set_log_levels
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor, LQGBTReductor, BRBTReductor
from pymor.reductors.h2 import IRKAReductor, TSIAReductor, OneSidedIRKAReductor
from pymor.reductors.mt import MTReductor


def fom_properties(fom, w, stable=True, fig_poles=None, fig_bode=None):
    """Show properties of the full-order model.

    Parameters
    ----------
    fom
        The full-order `Model` from :mod:`~pymor.models.iosys` or a |TransferFunction|.
    w
        Array of frequencies.
    stable
        Whether the FOM is stable.
    fig_poles
        Matplotlib figure for system poles.
    fig_bode
        Matplotlib figure for Bode plot.
    """
    from pymor.models.transfer_function import TransferFunction
    if not isinstance(fom, TransferFunction):
        print(f'order of the model = {fom.order}')
    print(f'number of inputs   = {fom.dim_input}')
    print(f'number of outputs  = {fom.dim_output}')

    # System norms
    if stable:
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

    # System poles
    if not isinstance(fom, TransferFunction):
        poles = fom.poles()
        if fig_poles is None:
            fig_poles = plt.figure()
        ax = fig_poles.subplots()
        ax.plot(poles.real, poles.imag, '.')
        ax.set_title('System poles')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imag')

        if not stable:
            ast_spectrum = fom.get_ast_spectrum()
            print(f'Anti-stable system poles:  {ast_spectrum[1]}')

    # Bode plot of the full model
    if fig_bode is None:
        fig_bode = plt.figure()
    ax = fig_bode.subplots(2 * fom.dim_output, fom.dim_input, squeeze=False)
    if isinstance(fom, TransferFunction):
        fom.bode_plot(w, ax=ax)
    else:
        fom.transfer_function.bode_plot(w, ax=ax)


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
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    subfigs1 = subfigs[1].subfigures(2, 1)
    fig.suptitle(f'{reductor_short_name} reduced-order model')

    # Bode plot of the full and reduced model
    axs = subfigs[0].subplots(2 * fom.dim_output, fom.dim_input, squeeze=False)
    if isinstance(fom, TransferFunction):
        fom.bode_plot(w, ax=axs, label='FOM')
    else:
        fom.transfer_function.bode_plot(w, ax=axs, label='FOM')
    rom.transfer_function.bode_plot(w, ax=axs, linestyle='dashed', label='ROM')
    for ax in axs.flat:
        ax.legend()

    # Poles of the reduced-order model
    poles_rom = rom.poles()
    ax = subfigs1[0].subplots()
    ax.plot(poles_rom.real, poles_rom.imag, '.')
    ax.set_title("ROM's poles")
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')

    # Magnitude plot of the error system
    ax = subfigs1[1].subplots()
    if isinstance(err, TransferFunction):
        err.mag_plot(w, ax=ax)
    else:
        err.transfer_function.mag_plot(w, ax=ax)
    ax.set_title('Magnitude plot of the error system')
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
            -\nabla z(0, x, t) \cdot n &= z(0, x, t),        & 0 < x < 1, t > 0 \\
            -\nabla z(1, x, t) \cdot n &= z(1, x, t),        & 0 < x < 1, t > 0 \\
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
    lti = fom.to_lti().with_(solver_options=solver_options)

    # Figure
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    subfigs1 = subfigs[1].subfigures(2, 1)
    fig.suptitle('Full-order model')

    # System properties
    w = (1e-1, 1e3)
    fom_properties(lti, w, fig_poles=subfigs1[0], fig_bode=subfigs[0])

    # Hankel singular values
    hsv = lti.hsv()
    ax = subfigs1[1].subplots()
    ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
    ax.set_title('Hankel singular values')
    ax.set_xlabel('Index')
    plt.show()

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

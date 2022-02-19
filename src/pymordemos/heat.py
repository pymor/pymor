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


def fom_properties(fom, w):
    """Show properties of the full-order model.

    Parameters
    ----------
    fom
        The full-order `Model` from :mod:`~pymor.models.iosys`.
    w
        Array of frequencies.
    """
    from pymor.models.transfer_function import TransferFunction
    if not isinstance(fom, TransferFunction):
        print(f'order of the model = {fom.order}')
    print(f'number of inputs   = {fom.dim_input}')
    print(f'number of outputs  = {fom.dim_output}')

    # System norms
    print(f'FOM H_2-norm:    {fom.h2_norm():e}')
    if not isinstance(fom, TransferFunction):
        if config.HAVE_SLYCOT:
            print(f'FOM H_inf-norm:  {fom.hinf_norm():e}')
        else:
            print('Skipped H_inf-norm calculation due to missing slycot.')
        print(f'FOM Hankel-norm: {fom.hankel_norm():e}')

    # System poles
    if not isinstance(fom, TransferFunction):
        poles = fom.poles()
        fig, ax = plt.subplots()
        ax.plot(poles.real, poles.imag, '.')
        ax.set_title('System poles')

    # Bode plot of the full model
    fig, ax = plt.subplots(2 * fom.dim_output, fom.dim_input, squeeze=False)
    if isinstance(fom, TransferFunction):
        fom.bode_plot(w, ax=ax)
    else:
        fom.transfer_function.bode_plot(w, ax=ax)
    fig.suptitle('Bode plot of the full model')
    plt.show()


def run_mor_method(lti, w, reductor, reductor_short_name, r, **reduce_kwargs):
    """Run a model order reduction method.

    Parameters
    ----------
    lti
        The full-order `Model` from :mod:`~pymor.models.iosys`.
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
    if isinstance(err, LTIModel):
        solver_options = {'lyap_lrcf': lyap_lrcf_solver_options(lradi_shifts='projection_shifts')['lradi']}
        err = err.with_(solver_options=solver_options)

    # Errors
    from pymor.models.transfer_function import TransferFunction
    if not isinstance(lti, TransferFunction):
        print(f'{reductor_short_name} relative H_2-error:    {err.h2_norm() / lti.h2_norm():e}')
        if config.HAVE_SLYCOT:
            print(f'{reductor_short_name} relative H_inf-error:  {err.hinf_norm() / lti.hinf_norm():e}')
        else:
            print('Skipped H_inf-norm calculation due to missing slycot.')
        print(f'{reductor_short_name} relative Hankel-error: {err.hankel_norm() / lti.hankel_norm():e}')
    elif isinstance(rom, LTIModel):
        error = np.sqrt(lti.h2_norm()**2 - 2 * lti.h2_inner(rom).real + rom.h2_norm()**2)
        print(f'{reductor_short_name} relative H_2-error:    {error / lti.h2_norm():e}')
    else:
        print(f'{reductor_short_name} relative H_2-error:    {err.h2_norm() / lti.h2_norm():e}')

    # Poles of the reduced-order model
    poles_rom = rom.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom.real, poles_rom.imag, '.')
    ax.set_title(f"{reductor_short_name} reduced model's poles")

    # Bode plot of the full and reduced model
    fig, ax = plt.subplots(2 * lti.dim_output, lti.dim_input, squeeze=False)
    if isinstance(lti, TransferFunction):
        lti.bode_plot(w, ax=ax)
    else:
        lti.transfer_function.bode_plot(w, ax=ax)
    rom.transfer_function.bode_plot(w, ax=ax, linestyle='dashed')
    fig.suptitle(f'Bode plot of the full and {reductor_short_name} reduced model')

    # Magnitude plot of the error system
    fig, ax = plt.subplots()
    if isinstance(err, TransferFunction):
        err.mag_plot(w, ax=ax)
    else:
        err.transfer_function.mag_plot(w, ax=ax)
    ax.set_title(f'Magnitude plot of the {reductor_short_name} error system')
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
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

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

    # System properties
    w = np.logspace(-1, 3, 100)
    fom_properties(lti, w)

    # Hankel singular values
    hsv = lti.hsv()
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
    ax.set_title('Hankel singular values')
    plt.show()

    # Model order reduction
    run_mor_method(lti, w, BTReductor(lti), 'BT', r, tol=1e-5)
    run_mor_method(lti, w, LQGBTReductor(lti), 'LQGBT', r, tol=1e-5)
    run_mor_method(lti, w, BRBTReductor(lti), 'BRBT', r, tol=1e-5)
    run_mor_method(lti, w, IRKAReductor(lti), 'IRKA', r)
    run_mor_method(lti, w, IRKAReductor(lti), 'IRKA (with Arnoldi)', r, projection='arnoldi')
    run_mor_method(lti, w, TSIAReductor(lti), 'TSIA', r)
    run_mor_method(lti, w, OneSidedIRKAReductor(lti, 'V'), 'OS-IRKA', r)
    run_mor_method(lti, w, MTReductor(lti), 'MT', r)


if __name__ == '__main__':
    run(main)

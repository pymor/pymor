#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import matplotlib.colors as cm
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.analyticalproblems.domaindescriptions import LineDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.core.logger import set_log_levels
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.aaa import PAAAReductor


def run_mor_method_dd_param(fom, ss, pp, reductor_cls, reductor_short_name, **reductor_kwargs):
    """Plot reductor for a sample parameter.

    Parameters
    ----------
    fom
        The full-order |LTIModel|.
    ss
        Transfer function sampling values.
    pp
        Parameter sampling values.
    reductor_cls
        The reductor class.
    reductor_short_name
        A short name for the reductor.
    reductor_kwargs
        Optional keyword arguments for the reductor class.
    """
    # Reduction
    rom = reductor_cls([ss * 1j, pp], fom, **reductor_kwargs).reduce()
    err = fom - rom

    n_w = 50
    n_p = 50
    w = np.geomspace(ss[0]/2, ss[-1]*2, n_w)
    p = np.geomspace(pp[0]/2, pp[-1]*2, n_p)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axs[0]
    sample_mu = np.median(pp)
    fom.transfer_function.mag_plot(w, ax=ax, mu=sample_mu, label='FOM')
    rom.mag_plot(w, ax=ax, mu=sample_mu, label='ROM', linestyle='dashed')
    err.mag_plot(w, ax=ax, mu=sample_mu, label='Error', linestyle='dotted')
    ax.set_title(fr'Magnitude plot for {reductor_short_name} with $\mu = {sample_mu}$')
    ax.legend()

    ax = axs[1]
    C = np.zeros((n_p, n_w))
    for i, mu in enumerate(p):
        C[i] = spla.norm(err.freq_resp(w, mu=mu), axis=(1, 2))
    out = ax.pcolormesh(w, p, C, shading='gouraud', norm=cm.LogNorm())
    ax.plot(*np.meshgrid(ss, pp, indexing='ij'), 'r.')
    ax.set(
        title=f'Magnitude plot for {reductor_short_name} error system',
        xlabel='Frequency (rad/s)',
        ylabel='Parameter',
        xscale='log',
        yscale='log',
    )
    fig.colorbar(out)

    plt.show()


def main(
        diameter: float = Argument(0.01, help='Diameter option for the domain discretizer.'),
        n: int = Argument(50, help='Number of frequency samples.'),
        m: int = Argument(10, help='Number of parameter samples.'),
):
    """Parametric 1D heat equation example."""
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

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

    lti = fom.to_lti()

    ss = np.logspace(-1, 4, n)
    pp = np.linspace(10, 100, m)

    run_mor_method_dd_param(lti, ss, pp, PAAAReductor, 'p-AAA')


if __name__ == "__main__":
    run(main)

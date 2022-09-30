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
from pymor.core.logger import set_log_levels
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.aaa import pAAAReductor


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
    rom = reductor_cls([ss, pp], fom, **reductor_kwargs).reduce()

    fig, ax = plt.subplots()
    sample_mu = np.median(pp)
    (fom - rom).mag_plot(ss, ax=ax, mu=sample_mu, label=fr'$\mu = {sample_mu}$')
    fom.transfer_function.mag_plot(ss, ax=ax, mu=sample_mu, label=fr'$\mu = {sample_mu}$')
    rom.mag_plot(ss, ax=ax, mu=sample_mu, label=fr'$\mu = {sample_mu}$')
    ax.set_title(f'Magnitude plot of the {reductor_short_name} error system')
    ax.legend()
    plt.show()


def main(
        diameter: float = Argument(0.01, help='Diameter option for the domain discretizer.'),
        n: int = Argument(50, help='Number of frequency samples.'),
        m: int = Argument(10, help='Number of parameter samples.'),
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

    lti = fom.to_lti()

    ss = np.logspace(-1, 4, n)
    pp = np.linspace(10, 100, m)

    run_mor_method_dd_param(lti, ss, pp, pAAAReductor, 'p-AAA')


if __name__ == "__main__":
    run(main)

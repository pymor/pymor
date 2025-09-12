# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import matplotlib.pyplot as plt
import numpy as np
from typer import Argument, run

from pymor.core.logger import set_log_levels
from pymor.models.examples import heat_equation_non_parametric_example
from pymor.models.transfer_function import TransferFunction
from pymor.reductors.aaa import PAAAReductor
from pymor.reductors.h2 import VectorFittingReductor
from pymor.reductors.loewner import LoewnerReductor


def run_mor_method_dd(fom, ss, reductor_cls, reductor_short_name, reductor_kwargs={}, reduce_args=(), reduce_kwargs={}):
    """Plot data-driven reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel|.
    ss
        Transfer function sampling values.
    reductor_cls
        The reductor class.
    reductor_short_name
        A short name for the reductor.
    reductor_kwargs
        Optional keyword arguments for the reductor class.
    reduce_args
        Optional positional arguments for the reduce method.
    reduce_kwargs
        Optional keyword arguments for the reduce method.
    """
    rom = reductor_cls(ss * 1j, fom, **reductor_kwargs).reduce(*reduce_args, **reduce_kwargs)
    err = fom - rom

    n_w = 50
    w = np.geomspace(ss[0]/2, ss[-1]*2, n_w)

    fig, ax = plt.subplots(constrained_layout=True)
    fom.transfer_function.mag_plot(w, ax=ax, label='FOM')
    if isinstance(rom, TransferFunction):
        rom.mag_plot(w, ax=ax, label='ROM', linestyle='dashed')
        err.mag_plot(w, ax=ax, label='Error', linestyle='dotted')
    else:
        rom.transfer_function.mag_plot(w, ax=ax, label='ROM', linestyle='dashed')
        err.transfer_function.mag_plot(w, ax=ax, label='Error', linestyle='dotted')
    ax.set_title(fr'Magnitude plot for {reductor_short_name}')
    ax.legend()


def main(
        diameter: float = Argument(0.1, help='Diameter option for the domain discretizer.'),
        n: int = Argument(50, help='Number of frequency samples.')
):
    """1D heat equation example."""
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

    fom = heat_equation_non_parametric_example(diameter=diameter)
    lti = fom.to_lti()

    ss = np.logspace(-1, 4, n)

    run_mor_method_dd(lti, ss, PAAAReductor, 'AAA')
    run_mor_method_dd(lti, ss, LoewnerReductor, 'Loewner')
    run_mor_method_dd(lti, ss, VectorFittingReductor, 'VF', reduce_args=(20,), reduce_kwargs={'maxit': 30})

    plt.show()


if __name__ == '__main__':
    run(main)

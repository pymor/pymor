#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from functools import partial
from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt
from typer import Argument, run

from pymor.models.examples import msd_example
from pymor.models.iosys import PHLTIModel
from pymor.reductors.bt import BTReductor, PRBTReductor
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.ph.ph_irka import PHIRKAReductor
from pymor.reductors.spectral_factor import SpectralFactorReductor


def main(
        n: int = Argument(100, help='Order of the mass-spring-damper system.'),
        m: int = Argument(2, help='Number of inputs and outputs of the mass-spring-damper system.'),
        max_reduced_order: int = Argument(20, help=('The maximum reduced order (at least 2). '
                                                    'Every even order below is used.')),
):
    J, R, G, P, S, N, E, Q = msd_example(n, m)

    # tolerance for solving the Riccati equation instead of KYP-LMI
    # by introducing a regularization feedthrough term D
    eps = 1e-12
    S += np.eye(S.shape[0]) * eps

    fom = PHLTIModel.from_matrices(J, R, G, S=S, Q=Q, solver_options={'ricc_pos_lrcf': 'slycot'})

    bt = BTReductor(fom).reduce
    prbt = PRBTReductor(fom).reduce
    irka = partial(IRKAReductor(fom).reduce, conv_crit='h2')
    phirka = PHIRKAReductor(fom).reduce
    spectral_factor = SpectralFactorReductor(fom)
    def spectral_factor_reduce(r):
        return spectral_factor.reduce(
            lambda spectral_factor, mu : IRKAReductor(spectral_factor,mu).reduce(r))

    reductors = {
        'BT': bt,
        'PRBT': prbt,
        'IRKA': irka,
        'pH-IRKA': phirka,
        'spectral_factor': spectral_factor_reduce,
    }
    markers = {
        'BT': '.',
        'PRBT': 'x',
        'IRKA': 'o',
        'pH-IRKA': 's',
        'spectral_factor': 'v',
    }
    timings = {}

    reduced_order = range(2, max_reduced_order + 1, 2)
    h2_errors = np.zeros((len(reductors), len(reduced_order)))

    for i, name in enumerate(reductors):
        t0 = perf_counter()
        for j, r in enumerate(reduced_order):
            rom = reductors[name](r)
            h2_errors[i, j] = (rom - fom).h2_norm() / fom.h2_norm()
        t1 = perf_counter()
        timings[name] = t1 - t0

    print('Timings:')
    for name, time in timings.items():
        print(f'  {name}: {time:.2f}s')

    fig, ax = plt.subplots()
    for i, reductor_name in enumerate(reductors):
        ax.semilogy(reduced_order, h2_errors[i], label=reductor_name, marker=markers[reductor_name])
    ax.set_xlabel('Reduced order $r$')
    ax.set_ylabel('Relative $\\mathcal{H}_2$-error')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    run(main)

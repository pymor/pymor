# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from functools import partial
from time import perf_counter

import numpy as np
from cyclopts import App
from matplotlib import pyplot as plt

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.examples import msd_example
from pymor.models.iosys import PHLTIModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.bt import BTReductor, PRBTReductor
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.ph.ph_irka import PHIRKAReductor
from pymor.reductors.spectral_factor import SpectralFactorReductor

app = App(help_on_error=True)

def ph_properties(model):
    """Check the three defining properties of a pH system."""
    densify = lambda op: to_matrix(op, format='dense')
    J, R, G, P, S, N, E, Q = map(densify, (model.J, model.R, model.G, model.P, model.S, model.N, model.E, model.Q))
    H = Q.T @ E
    Gamma = np.block([[J, G], [-G.T, N]])
    W = np.block([[R, P], [P.T, S]])

    assert np.allclose(np.abs(H - H.T).max(), 0), 'H is not symmetric'
    assert np.all(np.linalg.eigvalsh((H + H.T) / 2).min() > -1e-10), 'H is not positive definite'
    assert np.allclose(np.abs(Gamma + Gamma.T).max(), 0), 'Gamma is not symmetric'
    assert np.all(np.linalg.eigvalsh((W + W.T) / 2).min() > -1e-4), 'W is not positive semidefinite'


@app.default
def main(n: int = 100, m: int = 2, max_reduced_order: int = 20):
    """MOR for a port-Hamiltonian system.

    Parameters
    ----------
    n
        Order of the mass-spring-damper system.
    m
        Number of inputs and outputs of the mass-spring-damper system.
    max_reduced_order
        The maximum reduced order (at least 2). Every even order below is used.
    """
    fom = msd_example(n, m)

    # tolerance for solving the Riccati equation instead of KYP-LMI
    # by introducing a regularization feedthrough term D
    eps = 1e-12
    S = fom.S.matrix.copy()
    S += np.eye(S.shape[0]) * eps

    fom = fom.with_(S=NumpyMatrixOperator(S), N=None)

    bt = BTReductor(fom).reduce
    prbt = PRBTReductor(fom).reduce
    irka = partial(IRKAReductor(fom).reduce, conv_crit='h2')
    phirka_energy_stable = partial(PHIRKAReductor(fom).reduce, pg_projection='energy_stable')
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
        'pH-IRKA_energy_stable': phirka_energy_stable,
        'spectral_factor': spectral_factor_reduce,
    }

    ph_reductors = ['PRBT', 'pH-IRKA', 'spectral_factor']

    markers = {
        'BT': '.',
        'PRBT': 'x',
        'IRKA': 'o',
        'pH-IRKA': 's',
        'pH-IRKA_energy_stable': 'p',
        'spectral_factor': 'v',
    }
    timings = {}

    reduced_order = range(2, max_reduced_order + 1, 2)
    h2_errors = np.zeros((len(reductors), len(reduced_order)))

    for i, name in enumerate(reductors):
        t0 = perf_counter()
        for j, r in enumerate(reduced_order):
            rom = reductors[name](r)

            if name in ('PRBT', 'spectral_factor'):
               print('Converting ROM to PHLTIModel.')
               rom = PHLTIModel.from_passive_LTIModel(rom)

            if name in ph_reductors:
                print(name, r)
                ph_properties(rom)

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
    app()

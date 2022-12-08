#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import matplotlib.pyplot as plt
import numpy as np
from typer import Argument, run

from pymor.core.logger import set_log_levels
from pymor.models.transfer_function import TransferFunction
from pymor.reductors.h2 import TFIRKAReductor
from pymordemos.parametric_heat import fom_properties_param, run_mor_method_param


def main(r: int = Argument(10, help='Order of the TF-IRKA ROM.')):
    """Parametric delay demo."""
    set_log_levels({
        'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING',
        'pymor.algorithms.lradi.solve_lyap_lrcf': 'WARNING',
        'pymor.reductors.basic.LTIPGReductor': 'WARNING',
    })
    plt.rcParams['axes.grid'] = True

    # Model
    def H(s, mu):
        tau = mu['tau'][0]
        return np.array([[np.exp(-s) / (tau * s + 1)]])

    def dH(s, mu):
        tau = mu['tau'][0]
        return np.array([[-(tau * s + tau + 1) * np.exp(-s) / (tau * s + 1) ** 2]])

    fom = TransferFunction(1, 1,
                           H, dH,
                           parameters={'tau': 1})

    mus = [0.01, 0.1, 1]
    w = (1e-2, 1e4)
    fom_properties_param(fom, w, mus)

    # TF-IRKA
    run_mor_method_param(fom, r, w, mus, TFIRKAReductor, 'TF-IRKA')


if __name__ == "__main__":
    run(main)

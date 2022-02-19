#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import Argument, run

from pymor.models.transfer_function import TransferFunction
from pymor.reductors.h2 import TFIRKAReductor
from pymordemos.heat import fom_properties, run_mor_method


def main(
        tau: float = Argument(0.1, help='The time delay.'),
        r: int = Argument(10, help='Order of the TF-IRKA ROM.'),
):
    """Delay demo.

    Cascade of delay and integrator
    """
    # Transfer function
    def H(s):
        return np.array([[np.exp(-s) / (tau * s + 1)]])

    def dH(s):
        return np.array([[-(tau * s + tau + 1) * np.exp(-s) / (tau * s + 1) ** 2]])

    tf = TransferFunction(1, 1, H, dH)
    w = np.logspace(-1, 3, 500)
    fom_properties(tf, w)

    # Transfer function IRKA (TF-IRKA)
    run_mor_method(tf, w, TFIRKAReductor(tf), 'TF-IRKA', r, maxit=1000)


if __name__ == '__main__':
    run(main)

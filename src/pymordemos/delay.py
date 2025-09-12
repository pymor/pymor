# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.models.examples import transfer_function_delay_example
from pymor.reductors.h2 import TFIRKAReductor
from pymordemos.heat import fom_properties, run_mor_method


def main(
        tau: float = Argument(1, help='Time delay.'),
        a: float = Argument(-0.1, help='Pole without delay.'),
        r: int = Argument(8, help='Order of the TF-IRKA ROM.'),
):
    """Delay demo.

    Full-order model as a transfer function `exp(-tau*s) / (s - a)`.
    """
    plt.rcParams['axes.grid'] = True

    # Transfer function
    tf = transfer_function_delay_example(tau=tau, a=a)

    # Bode plot
    w = (1e-4, 1e2)
    fom_properties(tf, w)

    # Transfer function IRKA (TF-IRKA)
    run_mor_method(tf, w, TFIRKAReductor(tf), 'TF-IRKA', r, maxit=1000)


if __name__ == '__main__':
    run(main)

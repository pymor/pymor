# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import matplotlib.pyplot as plt
from cyclopts import App

from pymor.models.examples import transfer_function_delay_example
from pymor.reductors.h2 import TFIRKAReductor
from pymordemos.heat import fom_properties, run_mor_method

app = App(help_on_error=True)

@app.default
def main(
    tau: float = 1,
    a: float = -0.1,
    r: int = 8,
):
    """Delay demo.

    Full-order model as a transfer function `exp(-tau*s) / (s - a)`.

    Parameters
    ----------
    tau
        Time delay.
    a
        Pole without delay.
    r
        Order of the TF-IRKA ROM.
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
    app()

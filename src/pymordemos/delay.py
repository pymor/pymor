#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


import matplotlib.pyplot as plt
import numpy as np
from typer import Argument, run

from pymor.models.iosys import TransferFunction
from pymor.reductors.h2 import TFIRKAReductor


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

    # Transfer function IRKA (TF-IRKA)
    tf_irka_reductor = TFIRKAReductor(tf)
    rom = tf_irka_reductor.reduce(r, maxit=1000)

    # Final interpolation points
    sigma_list = tf_irka_reductor.sigma_list
    fig, ax = plt.subplots()
    ax.plot(sigma_list[-1].real, sigma_list[-1].imag, '.')
    ax.set_title('Final interpolation points of TF-IRKA')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    plt.show()

    # Magnitude plots
    w = np.logspace(-1, 3, 200)

    fig, ax = plt.subplots()
    tf.mag_plot(w, ax=ax)
    rom.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Magnitude plots of the full and reduced model')
    plt.show()

    fig, ax = plt.subplots()
    (tf - rom).mag_plot(w, ax=ax)
    ax.set_title('Magnitude plots of the error system')
    plt.show()


if __name__ == '__main__':
    run(main)

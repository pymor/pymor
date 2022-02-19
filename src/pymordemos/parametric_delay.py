#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import matplotlib.pyplot as plt
import numpy as np
from typer import Argument, run

from pymor.models.transfer_function import TransferFunction
from pymor.reductors.h2 import TFIRKAReductor


def main(r: int = Argument(10, help='Order of the TF-IRKA ROM.')):
    """Parametric delay demo."""
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

    # Magnitude plot
    mu_list = [0.01, 0.1, 1]

    w = np.logspace(-2, 4, 100)
    fig, ax = plt.subplots()
    for mu in mu_list:
        fom.mag_plot(w, ax=ax, mu=mu, label=fr'$\tau = {mu}$')
    ax.legend()
    plt.show()

    # TF-IRKA
    roms_tf_irka = []
    for mu in mu_list:
        tf_irka = TFIRKAReductor(fom, mu=mu)
        rom = tf_irka.reduce(r, conv_crit='h2', maxit=1000, num_prev=5)
        roms_tf_irka.append(rom)

    fig, ax = plt.subplots()
    for mu, rom in zip(mu_list, roms_tf_irka):
        poles = rom.poles()
        ax.plot(poles.real, poles.imag, '.', label=fr'$\tau = {mu}$')
    ax.set_title("Poles of TF-IRKA's ROMs")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    for mu, rom in zip(mu_list, roms_tf_irka):
        rom.transfer_function.mag_plot(w, ax=ax, label=fr'$\tau = {mu}$')
    ax.set_title("Magnitude plot of TF-IRKA's ROMs")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    for mu, rom in zip(mu_list, roms_tf_irka):
        (fom - rom).mag_plot(w, ax=ax, mu=mu, label=fr'$\tau = {mu}$')
    ax.set_title("Magnitude plot of error systems")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run(main)

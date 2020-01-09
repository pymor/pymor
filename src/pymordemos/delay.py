#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Delay demo

Cascade of delay and integrator
"""

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

from pymor.models.iosys import TransferFunction
from pymor.reductors.interpolation import TFBHIReductor
from pymor.reductors.h2 import TFIRKAReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace

if __name__ == '__main__':
    tau = 0.1

    def H(s):
        return np.array([[np.exp(-s) / (tau * s + 1)]])

    def dH(s):
        return np.array([[-(tau * s + tau + 1) * np.exp(-s) / (tau * s + 1) ** 2]])

    tf = TransferFunction(NumpyVectorSpace(1), NumpyVectorSpace(1), H, dH)

    r = 10
    tf_irka_reductor = TFIRKAReductor(tf)
    rom = tf_irka_reductor.reduce(r, maxit=1000)

    sigma_list = tf_irka_reductor.sigma_list
    fig, ax = plt.subplots()
    ax.plot(sigma_list[-1].real, sigma_list[-1].imag, '.')
    ax.set_title('Final interpolation points of TF-IRKA')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    plt.show()

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

    # step response
    E = rom.E.matrix
    A = rom.A.matrix
    B = rom.B.matrix
    C = rom.C.matrix

    nt = 1000
    t = np.linspace(0, 4, nt)
    x_old = np.zeros(rom.order)
    y = np.zeros(nt)
    for i in range(1, nt):
        h = t[i] - t[i - 1]
        x_new = spla.solve(E - h / 2 * A, (E + h / 2 * A).dot(x_old) + h * B[:, 0])
        x_old = x_new
        y[i] = C.dot(x_new)[0]

    step_response = np.piecewise(t, [t < 1, t >= 1], [0, 1]) * (1 - np.exp(-(t - 1) / tau))
    fig, ax = plt.subplots()
    ax.plot(t, step_response, '-', t, y, '--')
    ax.set_title('Step responses of the full and reduced model')
    ax.set_xlabel('$t$')
    plt.show()

    # match steady state (add interpolation point at 0)
    sigma_ss = list(sigma_list[-1]) + [0]
    b_ss = tf.input_space.ones(r + 1)
    c_ss = tf.output_space.ones(r + 1)
    interp_reductor = TFBHIReductor(tf)
    rom_ss = interp_reductor.reduce(sigma_ss, b_ss, c_ss)

    # step response
    E_ss = rom_ss.E.matrix
    A_ss = rom_ss.A.matrix
    B_ss = rom_ss.B.matrix
    C_ss = rom_ss.C.matrix

    x_ss_old = np.zeros(rom_ss.order)
    y_ss = np.zeros(nt)
    for i in range(1, nt):
        h = t[i] - t[i - 1]
        x_ss_new = spla.solve(E_ss - h / 2 * A_ss, (E_ss + h / 2 * A_ss).dot(x_ss_old) + h * B_ss[:, 0])
        x_ss_old = x_ss_new
        y_ss[i] = C_ss.dot(x_ss_new)[0]

    fig, ax = plt.subplots()
    ax.plot(t, step_response, '-', t, y_ss, '--')
    ax.set_title('Step responses of the full and reduced model 2')
    ax.set_xlabel('$t$')
    plt.show()

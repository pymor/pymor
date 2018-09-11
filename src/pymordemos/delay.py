#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Delay demo

Cascade of delay and integrator
"""

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

from pymor.discretizations.iosys import TransferFunction
from pymor.reductors.tf import TFInterpReductor, TF_IRKAReductor

if __name__ == '__main__':
    tau = 0.1

    def H(s):
        return np.array([[np.exp(-s) / (tau * s + 1)]])

    def dH(s):
        return np.array([[-(tau * s + tau + 1) * np.exp(-s) / (tau * s + 1) ** 2]])

    tf = TransferFunction(1, 1, H, dH)

    w = np.logspace(-1, 3, 1000)
    tfw = tf.bode(w)

    r = 10
    sigma = np.logspace(-2, 2, r)
    b = np.ones((1, r))
    c = np.ones((1, r))
    tol = 1e-3
    maxit = 1000
    tf_irka_reductor = TF_IRKAReductor(tf)
    rom = tf_irka_reductor.reduce(r, sigma, b, c, tol, maxit)

    sigmas = tf_irka_reductor.sigmas
    fig, ax = plt.subplots()
    ax.plot(sigmas[-1].real, sigmas[-1].imag, '.')
    ax.set_title('Final interpolation points of TF-IRKA')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')

    tfw_rom = rom.bode(w)
    fig, ax = plt.subplots()
    ax.loglog(w, np.abs(tfw[0, 0, :]), w, np.abs(tfw_rom[0, 0, :]))
    ax.set_title('Magnitude Bode plots of the full and reduced model')
    ax.set_xlabel(r'$\omega$')

    # step response
    E = rom.E.matrix
    A = rom.A.matrix
    B = rom.B.matrix
    C = rom.C.matrix

    nt = 1000
    t = np.linspace(0, 4, nt)
    x_old = np.zeros(rom.n)
    y = np.zeros(nt)
    for i in range(1, nt):
        h = t[i] - t[i - 1]
        x_new = spla.solve(E - h / 2 * A, (E + h / 2 * A).dot(x_old) + h * B[:, 0])
        x_old = x_new
        y[i] = C.dot(x_new)[0]

    step_response = np.piecewise(t, [t < 1, t >= 1], [0, 1]) * (1 - np.exp(-(t - 1) / tau))
    fig, ax = plt.subplots()
    ax.plot(t, step_response, 'b-', t, y, 'r-')
    ax.set_title('Step responses of the full and reduced model')
    ax.set_xlabel(r'$t$')
    plt.show()

    # match steady state (add interpolation point at 0)
    sigma_ss = list(sigmas[-1]) + [0]
    b_ss = np.ones((1, r + 1))
    c_ss = np.ones((1, r + 1))
    interp_reductor = TFInterpReductor(tf)
    rom_ss = interp_reductor.reduce(sigma_ss, b_ss, c_ss)

    # step response
    E_ss = rom_ss.E.matrix
    A_ss = rom_ss.A.matrix
    B_ss = rom_ss.B.matrix
    C_ss = rom_ss.C.matrix

    x_ss_old = np.zeros(rom_ss.n)
    y_ss = np.zeros(nt)
    for i in range(1, nt):
        h = t[i] - t[i - 1]
        x_ss_new = spla.solve(E_ss - h / 2 * A_ss, (E_ss + h / 2 * A_ss).dot(x_ss_old) + h * B_ss[:, 0])
        x_ss_old = x_ss_new
        y_ss[i] = C_ss.dot(x_ss_new)[0]

    fig, ax = plt.subplots()
    ax.plot(t, step_response, 'b-', t, y_ss, 'r-')
    ax.set_title('Step responses of the full and reduced model 2')
    ax.set_xlabel(r'$t$')
    plt.show()

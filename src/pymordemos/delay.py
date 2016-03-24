# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

import pymor.discretizations.iosys as iosys


tau = 1


def H(s):
    return np.exp(-s) / (tau * s + 1)


def dH(s):
    return -(tau * s + tau + 1) * np.exp(-s) / (tau * s + 1) ** 2

tf = iosys.TF(1, 1, H, dH)

w = np.logspace(-1, 3, 1000)
tfw = tf.bode(w)

r = 10
sigma = np.logspace(0, 1, r)
b = np.ones((1, r))
c = np.ones((1, r))
tol = 1e-4
maxit = 100
rom = tf.tf_irka(r, sigma, b, c, tol, maxit, verbose=True)

tfw_rom = rom.bode(w)
plt.loglog(w, np.abs(tfw[0, 0, :]), w, np.abs(tfw_rom[0, 0, :]))
plt.show()

# step response
E = rom.E._matrix
A = rom.A._matrix
B = rom.B._matrix
C = rom.C._matrix

nt = 1000
t = np.linspace(0, 4, nt)
x_old = np.zeros(rom.n, dtype=complex)
x_new = np.zeros(rom.n, dtype=complex)
y = np.zeros(nt, dtype=complex)
for i in xrange(1, nt):
    h = t[i] - t[i - 1]
    x_new = spla.solve(E - h / 2 * A, (E + h / 2 * A).dot(x_old) + h * B[:, 0])
    x_old = x_new
    y[i] = C.dot(x_new)[0]

step_response = np.piecewise(t, [t < 1, t >= 1], [0, 1]) * (1 - np.exp(-(t - 1) / tau))
plt.plot(t, step_response, 'b-', t, np.real(y), 'r-')
plt.show()

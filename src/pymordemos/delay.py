# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

import pymor.discretizations.iosys as iosys


def H(s):
    return np.exp(-s)

def dH(s):
    return -np.exp(-s)

tf = iosys.TF(1, 1, H, dH)

w = np.logspace(-3, 3, 1000)
tfw = tf.bode(w)

r = 10
sigma = np.logspace(-1, 0, r)
b = np.ones((1, r))
c = np.ones((1, r))
tol = 1e-4
maxit = 100
rom = tf.tf_irka(r, sigma, b, c, tol, maxit, verbose=True)

tfw_rom = rom.bode(w)

plt.loglog(w, np.abs(tfw[0, 0, :] - tfw_rom[0, 0, :]))
plt.show()

# step response
E = rom.E._matrix
A = rom.A._matrix
B = rom.B._matrix
C = rom.C._matrix

nt = 100
t = np.linspace(0, 5, nt)
x_old = np.zeros(rom.n, dtype=complex)
x_new = np.zeros(rom.n, dtype=complex)
y = np.zeros(nt, dtype=complex)
for i in xrange(1, nt):
    h = t[i] - t[i - 1]
    x_new = spla.solve(E - h * A, E.dot(x_old) + h * B[:, 0])
    y[i] = C.dot(x_new)[0]

plt.plot(t, y.real)
plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""1D heat equation demo

Discretization of the PDE::

    z_t(x, t) = z_xx(x, t)
    z_x(0, t) = z(0, t) - u(t)
    z_x(1, t) = -z(1, t)
    y(t) = z(1, t)

where u(t) is the input and y(t) is the output.
"""

from __future__ import absolute_import, division, print_function

import pymor.discretizations.iosys as iosys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # dimension of the system
    n = 100

    # assemble A, B, and C
    A = np.zeros((n, n))
    A[0, 0] = -2 * n * (n - 1)
    A[0, 1] = 2 * (n - 1) ** 2
    for i in xrange(1, n - 1):
        A[i, i - 1] = (n - 1) ** 2
        A[i, i] = -2 * (n - 1) ** 2
        A[i, i + 1] = (n - 1) ** 2
    A[n - 1, n - 1] = -2 * n * (n - 1)
    A[n - 1, n - 2] = 2 * (n - 1) ** 2

    B = np.zeros((n, 1))
    B[0, 0] = 2 * (n - 1)

    C = np.zeros((1, n))
    C[0, n - 1] = 1

    # eigenvalues of A
    #print(A)
    #ev = np.linalg.eigvals(A)
    #print(ev)
    #print(ev.real)
    #print(ev.imag)
    #fig, ax = plt.subplots()
    #ax.plot(ev.real, ev.imag, '.')
    #plt.show()

    # LTI system
    lti = iosys.LTISystem.from_matrices(A, B, C)

    print('n = {}'.format(lti.n))
    print('m = {}'.format(lti.m))
    print('p = {}'.format(lti.p))

    # Hankel singular values
    lti.compute_hsv_U_V()
    #print(lti._hsv)

    # H_2-norm of the system
    print(lti.norm())

    # Balanced Truncation
    r = 5
    rom_bt, _, _ = lti.bt(r)
    print(rom_bt.norm())
    err_bt = lti - rom_bt
    print(err_bt.norm())

    # Iterative Rational Krylov Algorithm
    sigma = np.logspace(-1, 3, r)
    np.random.seed(1)
    b = np.random.randn(lti.m, r)
    c = np.random.randn(lti.p, r)
    tol = 1e-4
    maxit = 100
    rom_irka, _, reduction_data_irka = lti.irka(sigma, b, c, tol, maxit, prnt=True)

    #print(reduction_data_irka['dist'])
    tmp = map(np.min, reduction_data_irka['dist'])
    print(tmp)
    #fig, ax = plt.subplots()
    #ax.semilogy(tmp)
    #plt.show()

    print(rom_irka.norm())
    err_irka = lti - rom_irka
    print(err_irka.norm())

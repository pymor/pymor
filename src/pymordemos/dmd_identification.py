#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import run

from pymor.algorithms.dmd import dmd
from pymor.algorithms.to_matrix import to_matrix
from pymor.basic import *


def main():
    N = 10
    state_dim = 4
    A = np.random.rand(state_dim, state_dim)
    A = A / np.linalg.norm(A)
    print(f'A: {A}')
    X = np.zeros((N, state_dim))
    x = np.ones(state_dim)

    X[0] = x

    for i in range(N - 1):
        x = A @ x
        X[i + 1] = x

    Xva = NumpyVectorSpace.from_numpy(X)

    Wk, omega, Aop = dmd(Xva, return_A_approx=True)

    A_approx = to_matrix(Aop)
    print(f'A_approx: {A_approx}')

    error = np.linalg.norm(A - A_approx, 'fro')
    print(f'Error |A-A_approx|_fro: {error:.2e}')

    tol = 1e-5

    assert error <= tol


if __name__ == '__main__':
    run(main)

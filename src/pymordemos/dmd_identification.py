#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from typer import run, Option

from pymor.algorithms.dmd import dmd
from pymor.basic import *


def main(
        n: int = Option(4, help='Dimension of the state.'),
        m: int = Option(10, help='Number of data pairs.')
):
    A = np.random.rand(n, n)
    A = A / np.linalg.norm(A)
    print(f'A: {A}')
    X = np.zeros((m + 1, n))
    x = np.ones(n)

    X[0] = x

    for i in range(m):
        x = A @ x
        X[i + 1] = x

    Xva = NumpyVectorSpace.from_numpy(X)

    Wk, omega, A_tilde, Uva = dmd(Xva, return_A_tilde=True)
    U = Uva.to_numpy()

    A_approx = U.conj().T @ A_tilde @ U
    print(f'A_approx: {A_approx}')

    error = np.linalg.norm(A - A_approx, 'fro')
    print(f'Error |A-A_approx|_fro: {error:.2e}')

    tol = 1e-5

    assert error <= tol


if __name__ == '__main__':
    run(main)

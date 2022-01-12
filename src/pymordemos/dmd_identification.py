#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from typer import run, Option

from pymor.algorithms.to_matrix import to_matrix
from pymor.basic import *


def main(
        n: int = Option(4, help='Dimension of the state.'),
        M: int = Option(10, help='Number of data pairs.'),
        seed: int = Option(42, help='Random seed.')
):
    np.random.seed(seed)

    A = np.random.rand(n, n)
    A = A / np.linalg.norm(A)
    print(f'A: {A}')
    X = np.zeros((M + 1, n))
    x = np.ones(n)

    X[0] = x

    for i in range(M):
        x = A @ x
        X[i + 1] = x

    Xva = NumpyVectorSpace.from_numpy(X)

    Wk, omega, A_approx = dmd(Xva, return_A_approx=True)

    A_approx = to_matrix(A_approx)
    print(f'A_approx: {A_approx}')

    error = np.linalg.norm(A - A_approx, 'fro')
    print(f'Error |A-A_approx|_fro: {error:.2e}')


if __name__ == '__main__':
    run(main)

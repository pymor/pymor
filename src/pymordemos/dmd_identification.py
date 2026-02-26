# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from cyclopts import App

from pymor.algorithms.to_matrix import to_matrix
from pymor.basic import *

app = App(help_on_error=True)

@app.default
def main(
    n: int = 4,
    M: int = 10,
):
    """DMD system identification demo.

    Parameters
    ----------
    n
        Dimension of the state.
    M
        Number of data pairs.
    """
    A = get_rng().random((n, n))
    A = A / np.linalg.norm(A)
    print(f'A: {A}')
    X = np.zeros((n, M + 1))
    x = np.ones(n)

    X[:, 0] = x

    for i in range(M):
        x = A @ x
        X[:, i + 1] = x

    Xva = NumpyVectorSpace.from_numpy(X)

    Wk, omega, A_approx = dmd(Xva, return_A_approx=True)

    A_approx = to_matrix(A_approx)
    print(f'A_approx: {A_approx}')

    error = np.linalg.norm(A - A_approx, 'fro')
    print(f'Error |A-A_approx|_fro: {error:.2e}')


if __name__ == '__main__':
    app()

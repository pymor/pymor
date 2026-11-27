# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from cyclopts import App

from pymor.algorithms.to_matrix import to_matrix
from pymor.basic import *

TAGS = ('instationary', 'linear', 'data-driven', 'system-identification')

app = App(help_on_error=True)

@app.default
def main(
    n: int = 4,
    M: int = 10,
):
    r"""DMD system identification demo.

    This demo identifies a linear discrete-time system :math:`x_{k+1} = A x_k` from trajectory
    data using dynamic mode decomposition (DMD). The system matrix :math:`A` is a random
    :math:`n \times n` matrix normalized to Frobenius norm one, and the trajectory is generated
    from the initial state :math:`x_0 = (1, \dots, 1)^\top`. The DMD
    (:func:`~pymor.algorithms.dmd.dmd`, :cite:`TRLBK14`) of the trajectory yields a low-rank
    approximation of :math:`A`, and the original matrix, its approximation and the error in the
    Frobenius norm are printed. If the trajectory spans the whole state space, :math:`A` is
    recovered up to rounding errors.

    For the DMD of a nonlinear system, see :ref:`cyclopts-pymor-demo-burgers_dmd`, and for
    system identification from impulse response data using the eigensystem realization
    algorithm, see :ref:`cyclopts-pymor-demo-era`.

    Parameters
    ----------
    n
        Dimension of the state.
    M
        Number of data pairs (time steps of the trajectory).
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

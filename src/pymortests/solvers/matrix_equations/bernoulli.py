# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla
from scipy.stats import ortho_group

from pymor.solvers.matrix_equations.equations import BernoulliEquation

pytestmark = pytest.mark.builtin

n_list = [10, 20, 30]


def residual(A, E, B, X, trans):
    if not trans:
        return spla.norm(A @ X @ E.conj().T + E @ X @ A.conj().T
                         - E @ X @ B.conj().T @ B @ X @ E.conj().T) / spla.norm(X)
    else:
        return spla.norm(A.conj().T @ X @ E + E.conj().T @ X @ A
                         - E.conj().T @ X @ B @ B.conj().T @ X @ E) / spla.norm(X)


def bernoulli_matrices(n, trans, rng):
    E = -ortho_group.rvs(dim=n, random_state=0)
    A = np.diag(np.concatenate((np.arange(-n + 4, 0), np.arange(1, 5)))) @ E
    A = A + 1.j * A
    B = rng.standard_normal((n, 1))

    if not trans:
        B = B.conj().T

    return A, E, B


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('trans', [False, True])
def test_bernoulli(n, trans, rng):
    A, E, B = bernoulli_matrices(n, trans, rng)

    X = BernoulliEquation.from_matrices(A, E, B, trans=trans).solve()

    assert residual(A, E, B, X, trans) < 1e-9


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('trans', [False, True])
def test_bernoulli_lr(n, trans, rng):
    A, E, B = bernoulli_matrices(n, trans, rng)

    Zva = BernoulliEquation.from_matrices(A, E, B, trans=trans).solve_lr()
    Z = Zva.to_numpy()
    X = Z @ Z.conj().T

    assert residual(A, E, B, X, trans) < 1e-9

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
from scipy.stats import ortho_group

from pymor.algorithms.bernoulli import bernoulli_stabilize, solve_bernoulli
from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.constructions import LowRankOperator
from pymor.operators.numpy import NumpyMatrixOperator

import pytest


n_list = [10, 20, 30]


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('trans', [False, True])
def test_bernoulli(n, with_E, trans):
    np.random.seed(0)
    E = -ortho_group.rvs(dim=n)
    A = np.diag(np.concatenate((np.arange(-n + 4, 0), np.arange(1, 5)))) @ E
    A = A + 1.j * A
    B = np.random.randn(n, 1)

    if not trans:
        B = B.conj().T

    Yp = solve_bernoulli(A, E, B, trans=trans)
    X = Yp @ Yp.conj().T

    if not trans:
        assert spla.norm(A @ X @ E.conj().T + E @ X @ A.conj().T
                         - E @ X @ B.conj().T @ B @ X @ E.conj().T) / spla.norm(X) < 1e-9
    else:
        assert spla.norm(A.conj().T @ X @ E + E.conj().T @ X @ A
                         - E.conj().T @ X @ B @ B.conj().T @ X @ E) / spla.norm(X) < 1e-9


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('trans', [False, True])
def test_bernoulli_stabilize(n, trans):
    np.random.seed(0)
    A = sps.random(n, n, density=0.3)
    Aop = NumpyMatrixOperator(A)

    B = np.random.randn(1, n)
    if not trans:
        Bva = Aop.range.from_numpy(B)
    else:
        Bva = Aop.source.from_numpy(B)

    ew, lev, rev = spla.eig(A.todense(), None, True)
    as_idx = np.where(ew.real > 0.)
    lva = Aop.source.from_numpy(lev[:, as_idx][:, 0, :].T)
    rva = Aop.range.from_numpy(rev[:, as_idx][:, 0, :].T)

    K = bernoulli_stabilize(Aop, None, Bva, (lva, ew, rva), trans=trans)

    if not trans:
        A_stab = to_matrix(Aop - LowRankOperator(K, np.eye(len(Bva)), Bva))
    else:
        A_stab = to_matrix(Aop - LowRankOperator(Bva, np.eye(len(Bva)), K))

    ew, _ = spla.eig(A_stab)
    assert np.all(np.real(ew) <= 0)

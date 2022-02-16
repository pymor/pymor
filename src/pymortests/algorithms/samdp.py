# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.samdp import samdp
from pymor.operators.numpy import NumpyMatrixOperator

import pytest

n_list = [50, 100]
m_list = [1, 2]
k_list = [2, 3]
wanted_list = [15, 20]
which_list = ['NR', 'NS', 'NM']


def conv_diff_1d_fd(n, a, b):
    diagonals = [-a * 2 * (n + 1) ** 2 * np.ones((n,)),
                 (a * (n + 1) ** 2 + b * (n + 1) / 2) * np.ones((n - 1,)),
                 (a * (n + 1) ** 2 - b * (n + 1) / 2) * np.ones((n - 1,))]
    A = sps.diags(diagonals, [0, -1, 1], format='csc')
    return A


def conv_diff_1d_fem(n, a, b):
    diagonals = [-a * 2 * (n + 1) ** 2 * np.ones((n,)),
                 (a * (n + 1) ** 2 + b * (n + 1) / 2) * np.ones((n - 1,)),
                 (a * (n + 1) ** 2 - b * (n + 1) / 2) * np.ones((n - 1,))]
    A = sps.diags(diagonals, [0, -1, 1], format='csc')
    diagonals = [2 / 3 * np.ones((n,)),
                 1 / 6 * np.ones((n - 1,)),
                 1 / 6 * np.ones((n - 1,))]
    E = sps.diags(diagonals, [0, -1, 1], format='csc')
    return A, E


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('k', k_list)
@pytest.mark.parametrize('wanted', wanted_list)
@pytest.mark.parametrize('which', which_list)
@pytest.mark.parametrize('with_E', [False, True])
def test_samdp(n, m, k, wanted, with_E, which):
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        E = sps.eye(n)
        Eop = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        Eop = NumpyMatrixOperator(E)

    np.random.seed(0)
    B = np.random.randn(n, m)
    C = np.random.randn(k, n)

    Aop = NumpyMatrixOperator(A)
    Bva = Aop.source.from_numpy(B.T)
    Cva = Aop.source.from_numpy(C)

    dom_poles, dom_res, dom_rev, dom_lev = samdp(Aop, Eop, Bva, Cva, wanted, which=which)

    dom_absres = spla.norm(dom_res, ord=2, axis=(1, 2))

    poles, lev, rev = spla.eig(A.toarray(), E.toarray(), left=True)

    absres = np.empty(len(poles))

    for i in range(len(poles)):
        lev[:, i] = lev[:, i] * (1 / lev[:, i].conj().dot(E @ rev[:, i]))
        absres[i] = spla.norm(np.outer(C @ rev[:, i], lev[:, i] @ B), ord=2)

    if which == 'NR':
        val = absres / np.abs(np.real(poles))
        dom_val = dom_absres / np.abs(np.real(dom_poles))
    elif which == 'NS':
        val = absres / np.abs(poles)
        dom_val = dom_absres / np.abs(dom_poles)
    elif which == 'NM':
        val = absres
        dom_val = dom_absres

    # check if computed poles are approximately more dominant than others on average
    assert np.average(val) * 0.9 < np.average(dom_val)

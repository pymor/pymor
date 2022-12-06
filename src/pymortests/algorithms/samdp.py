# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
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

    B = np.random.randn(n, m)
    C = np.random.randn(k, n)

    Aop = NumpyMatrixOperator(A)
    Bva = Aop.source.from_numpy(B.T)
    Cva = Aop.source.from_numpy(C)

    dom_poles, dom_res, dom_rev, dom_lev = samdp(Aop, Eop, Bva, Cva, wanted, which=which)

    # check if we computed correct eigenvalues
    if not with_E:
        assert np.sum((Aop.apply(dom_rev) - dom_poles * dom_rev).norm()) < 1e-4
        assert np.sum((Aop.apply_adjoint(dom_lev) - dom_poles * dom_lev).norm()) < 1e-4
    else:
        assert np.sum((Aop.apply(dom_rev) - dom_poles * Eop.apply(dom_rev)).norm()) < 1e-4
        assert np.sum((Aop.apply_adjoint(dom_lev) - dom_poles * Eop.apply_adjoint(dom_lev)).norm()) < 1e-4

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.sparse as sps

from pymor.algorithms.eigs import eigs
from pymor.operators.numpy import NumpyMatrixOperator

n_list = [100, 200]
k_list = [1, 7]
which_list = ['LM', 'LR', 'LI']


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('k', k_list)
@pytest.mark.parametrize('which', which_list)
def test_eigs(n, k, which):
    np.random.seed(0)
    A = sps.random(n, n, density=0.1)
    Aop = NumpyMatrixOperator(A)
    ew, ev = eigs(Aop, k=k, which=which)

    assert np.sum((Aop.apply(ev) - ev * ew).l2_norm()) < 1e-4

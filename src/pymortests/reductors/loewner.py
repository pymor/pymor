# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.sparse as sps

from pymor.models.iosys import LTIModel
from pymor.reductors.loewner import LoewnerReductor

pytestmark = pytest.mark.builtin

custom_partitioning = np.random.permutation(10)
test_data = [
    ({'r': 3}, {}),
    ({'tol': 1e-3}, {}),
    ({'tol': 1e-3}, {'partitioning': 'even-odd'}),
    ({'tol': 1e-3}, {'partitioning': 'half-half'}),
    ({'tol': 1e-3}, {'partitioning': (custom_partitioning[:5], custom_partitioning[5:])}),
    ({'tol': 1e-3}, {'ordering': 'magnitude'}),
    ({'tol': 1e-3}, {'ordering': 'random'}),
    ({'tol': 1e-3}, {'ordering': 'regular'}),
    ({'tol': 1e-3}, {'conjugate': False}),
    ({'tol': 1e-3}, {'mimo_handling': 'full'}),
    ({'tol': 1e-3}, {'mimo_handling': 'random'}),
    ({'tol': 1e-3}, {'mimo_handling': (np.random.rand(10, 3), np.random.rand(2, 10))})
]


@pytest.mark.parametrize('reduce_kwargs,loewner_kwargs', test_data)
def test_loewner_lti(reduce_kwargs, loewner_kwargs):
    n = 10
    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.arange(2*n).reshape(n, 2)
    C = np.arange(3*n).reshape(3, n)
    s = np.logspace(-2, 2, 10)*1j
    fom = LTIModel.from_matrices(A, B, C)
    loewner = LoewnerReductor(s, fom, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert isinstance(rom, LTIModel)


@pytest.mark.parametrize('reduce_kwargs,loewner_kwargs', test_data)
def test_loewner_tf(reduce_kwargs, loewner_kwargs):
    n = 10
    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.arange(2*n).reshape(n, 2)
    C = np.arange(3*n).reshape(3, n)
    s = np.logspace(-2, 2, 10)*1j
    fom = LTIModel.from_matrices(A, B, C)
    loewner = LoewnerReductor(s, fom.transfer_function, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert isinstance(rom, LTIModel)


@pytest.mark.parametrize('reduce_kwargs,loewner_kwargs', test_data)
def test_loewner_data(reduce_kwargs, loewner_kwargs):
    n = 10
    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.arange(2*n).reshape(n, 2)
    C = np.arange(3*n).reshape(3, n)
    s = np.logspace(-2, 2, 10)
    fom = LTIModel.from_matrices(A, B, C)
    Hs = fom.transfer_function.freq_resp(s)
    loewner = LoewnerReductor(s*1j, Hs, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert isinstance(rom, LTIModel)

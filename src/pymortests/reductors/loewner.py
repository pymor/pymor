# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.sparse as sps

from pymor.models.iosys import LTIModel
from pymor.reductors.loewner import LoewnerReductor

pytestmark = pytest.mark.builtin


def custom_partitioning(rng):
    p = rng.permutation(40)
    return p[:20], p[20:]

@pytest.fixture(
    params=[
        lambda rng: ({'r': 20}, {}),
        lambda rng: ({'tol': 1e-12}, {}),
        lambda rng: ({'tol': 1e-12}, {'partitioning': 'even-odd'}),
        lambda rng: ({'tol': 1e-12}, {'partitioning': 'half-half'}),
        lambda rng: ({'tol': 1e-12}, {'partitioning': custom_partitioning(rng)}),
        lambda rng: ({'tol': 1e-12}, {'ordering': 'magnitude'}),
        lambda rng: ({'tol': 1e-12}, {'ordering': 'random'}),
        lambda rng: ({'tol': 1e-12}, {'ordering': 'regular'}),
        lambda rng: ({'tol': 1e-12}, {'conjugate': False}),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': 'full'}),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': 'random'}),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': (rng.random((40, 3)), rng.random((2, 40)))})
    ])
def reduce_kwargs_and_lowener_kwargs(rng, request):
    return request.param(rng)

@pytest.fixture()
def reduce_kwargs(reduce_kwargs_and_lowener_kwargs):
    return reduce_kwargs_and_lowener_kwargs[0]

@pytest.fixture()
def loewner_kwargs(reduce_kwargs_and_lowener_kwargs):
    return reduce_kwargs_and_lowener_kwargs[1]


def test_loewner_lti(reduce_kwargs, loewner_kwargs):
    fom = make_fom(10)
    s = np.logspace(1, 3, 40)*1j
    loewner = LoewnerReductor(s, fom, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])


def test_loewner_tf(reduce_kwargs, loewner_kwargs):
    fom = make_fom(10)
    s = np.logspace(1, 3, 40)*1j
    loewner = LoewnerReductor(s, fom.transfer_function, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])


def test_loewner_data(reduce_kwargs, loewner_kwargs):
    fom = make_fom(10)
    s = np.logspace(1, 3, 40)
    Hs = fom.transfer_function.freq_resp(s)
    loewner = LoewnerReductor(s*1j, Hs, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])


def make_fom(n):
    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.arange(2*n).reshape(n, 2)
    C = np.arange(3*n).reshape(3, n)
    return LTIModel.from_matrices(A, B, C)

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.examples import penzl_mimo_example
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
def reduce_kwargs_and_loewner_kwargs(rng, request):
    return request.param(rng)

@pytest.fixture
def reduce_kwargs(reduce_kwargs_and_loewner_kwargs):
    return reduce_kwargs_and_loewner_kwargs[0]

@pytest.fixture
def loewner_kwargs(reduce_kwargs_and_loewner_kwargs):
    return reduce_kwargs_and_loewner_kwargs[1]


def test_loewner_lti(reduce_kwargs, loewner_kwargs):
    fom = penzl_mimo_example(10)
    s = np.logspace(1, 3, 40)*1j
    loewner = LoewnerReductor(s, fom, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])


def test_loewner_tf(reduce_kwargs, loewner_kwargs):
    fom = penzl_mimo_example(10)
    s = np.logspace(1, 3, 40)*1j
    loewner = LoewnerReductor(s, fom.transfer_function, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])


def test_loewner_data(reduce_kwargs, loewner_kwargs):
    fom = penzl_mimo_example(10)
    s = np.logspace(1, 3, 40)
    Hs = fom.transfer_function.freq_resp(s)
    loewner = LoewnerReductor(s*1j, Hs, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])

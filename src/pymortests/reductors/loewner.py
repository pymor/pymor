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
        lambda rng: ({'r': 20}, {}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {}, [20, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {}, [10, 1, 3]),
        lambda rng: ({'tol': 1e-12}, {}, [10, 2, 1]),
        lambda rng: ({'tol': 1e-12}, {}, [10, 1, 1]),
        lambda rng: ({'tol': 1e-12}, {'partitioning': 'even-odd'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'partitioning': 'half-half'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'partitioning': custom_partitioning(rng)}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'ordering': 'magnitude'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'ordering': 'random'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'ordering': 'regular'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'conjugate': False}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': 'full'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': 'random'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': (rng.random((20, 3)), rng.random((2, 20))),
                                      'conjugate': False}, [10, 2, 3])
    ])
def reduce_kwargs_and_loewner_kwargs_and_model_args(rng, request):
    return request.param(rng)

@pytest.fixture
def reduce_kwargs(reduce_kwargs_and_loewner_kwargs_and_model_args):
    return reduce_kwargs_and_loewner_kwargs_and_model_args[0]

@pytest.fixture
def loewner_kwargs(reduce_kwargs_and_loewner_kwargs_and_model_args):
    return reduce_kwargs_and_loewner_kwargs_and_model_args[1]

@pytest.fixture
def model_args(reduce_kwargs_and_loewner_kwargs_and_model_args):
    return reduce_kwargs_and_loewner_kwargs_and_model_args[2]


def test_loewner_lti(reduce_kwargs, loewner_kwargs, model_args):
    fom = penzl_mimo_example(*model_args)
    s = np.logspace(1, 3, 40)*1j
    loewner = LoewnerReductor(s, fom, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])
    assert rom.order <= model_args[0]


def test_loewner_tf(reduce_kwargs, loewner_kwargs, model_args):
    fom = penzl_mimo_example(*model_args)
    s = np.logspace(1, 3, 40)*1j
    loewner = LoewnerReductor(s, fom.transfer_function, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])
    assert rom.order <= model_args[0]


def test_loewner_data(reduce_kwargs, loewner_kwargs, model_args):
    fom = penzl_mimo_example(*model_args)
    s = np.logspace(1, 3, 40)
    Hs = fom.transfer_function.freq_resp(s)
    loewner = LoewnerReductor(s*1j, Hs, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])
    assert rom.order <= model_args[0]

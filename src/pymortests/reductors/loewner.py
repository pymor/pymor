# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.loewner import _sample_transfer_function, partition_frequencies
from pymor.models.examples import penzl_mimo_example
from pymor.models.iosys import LTIModel
from pymor.reductors.aaa import PAAAReductor
from pymor.reductors.h2 import VectorFittingReductor
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
        lambda rng: ({'tol': 1e-12}, {'force_real': False}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': 'full'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': 'random'}, [10, 2, 3]),
        lambda rng: ({'tol': 1e-12}, {'mimo_handling': (rng.random((20, 3)), rng.random((2, 20))),
                                      'force_real': False}, [10, 2, 3])
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
    Hs = LoewnerReductor.generate_samples(s, fom)
    loewner = LoewnerReductor(s, Hs, **loewner_kwargs)
    rom = loewner.reduce(**reduce_kwargs)
    assert np.all([np.abs(fom.transfer_function.eval_tf(ss) - rom.transfer_function.eval_tf(ss))
        / np.abs(fom.transfer_function.eval_tf(ss)) < 1e-10 for ss in s])
    assert rom.order <= model_args[0]


def test_loewner_tf(reduce_kwargs, loewner_kwargs, model_args):
    fom = penzl_mimo_example(*model_args)
    s = np.logspace(1, 3, 40)*1j
    Hs = LoewnerReductor.generate_samples(s, fom.transfer_function)
    loewner = LoewnerReductor(s, Hs, **loewner_kwargs)
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


def test_loewner_unitary_realification():
    s = np.array([0, 2, 1j, -1j, 2 + 2j, 2 - 2j])
    Hs = np.array([sum((i + 1) / (ss + i + 1) for i in range(5)) for ss in s])
    partitioning = (np.array([0, 2, 3]), np.array([1, 4, 5]))

    complex_quadruple = LoewnerReductor(s, Hs, partitioning=partitioning,
                                       force_real=False).loewner_quadruple()
    real_quadruple = LoewnerReductor(s, Hs, force_real=True).loewner_quadruple()

    assert all(not np.iscomplexobj(matrix) for matrix in real_quadruple)
    for complex_matrix, real_matrix in zip(complex_quadruple, real_quadruple, strict=True):
        assert np.allclose(np.linalg.norm(complex_matrix), np.linalg.norm(real_matrix))


def test_loewner_completes_explicit_partitions():
    s = np.array([0, 1j, -1j, 2j, -3j, 2])
    Hs = 1 / (s + 1)
    partitioning = (np.array([3, 0]), np.array([5, 4, 2, 1]))

    reductor = LoewnerReductor(s, Hs, partitioning=partitioning)

    assert np.array_equal(reductor.s, [0, 1j, -1j, 2j, -3j, 2, -2j, 3j])
    assert np.array_equal(reductor.Hs, 1 / (reductor.s + 1))
    assert np.array_equal(reductor.partitioning[0], [3, 0, 6])
    assert np.array_equal(reductor.partitioning[1], [5, 4, 2, 1, 7])
    assert np.array_equal(partitioning[0], [3, 0])
    assert np.array_equal(partitioning[1], [5, 4, 2, 1])
    assert all(not np.iscomplexobj(matrix) for matrix in reductor.loewner_quadruple())


def test_partition_frequencies_magnitude_ordering_without_conjugates():
    s = 1j * np.arange(1, 5)
    Hs = np.array([4., 1., 3., 2.])

    left, right = partition_frequencies(s, Hs, ordering='magnitude', force_real=False)
    assert np.array_equal(left, [1, 2])
    assert np.array_equal(right, [3, 0])
    assert all(np.all(np.isfinite(matrix))
               for matrix in LoewnerReductor(s, Hs, ordering='magnitude', force_real=False).loewner_quadruple())


@pytest.mark.parametrize('reductor_cls', [LoewnerReductor, PAAAReductor, VectorFittingReductor])
@pytest.mark.parametrize('model_input', [False, True])
def test_data_driven_reductor_sampling_api(reductor_cls, model_input):
    fom = LTIModel.from_matrices(np.array([[-1.]]), np.ones((1, 1)), np.ones((1, 1)))
    source = fom if model_input else fom.transfer_function
    nodes = np.array([1j, 2j])
    with pytest.raises(AssertionError, match='generate_samples'):
        reductor_cls(nodes, source)

    assert reductor_cls.generate_samples is _sample_transfer_function
    samples = reductor_cls.generate_samples(nodes, source)
    derivatives = reductor_cls.generate_samples(nodes, source, derivative=True)
    assert np.allclose(samples[:, 0, 0], 1 / (nodes + 1))
    assert np.allclose(derivatives[:, 0, 0], -1 / (nodes + 1)**2)
    reductor = reductor_cls(nodes, samples, force_real=False)
    assert np.array_equal(reductor.generate_samples(nodes, source), samples)
    with pytest.raises(AssertionError, match='fom must be'):
        reductor_cls.generate_samples(nodes, samples)

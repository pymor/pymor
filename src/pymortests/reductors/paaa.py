# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import itertools

import numpy as np
import pytest

from pymor.reductors.aaa import PAAAReductor, full_nd_loewner

pytestmark = pytest.mark.builtin

test_data = [
    (3, 2, True),
    (3, 2, False),
    (1, 1, True),
    (1, 1, False),
    (3, 1, True),
    (3, 1, False),
    (1, 3, True),
    (1, 3, False)
]


@pytest.mark.parametrize(('m', 'p', 'is_parametric'), test_data)
def test_paaa(m,p,is_parametric, rng):
    if is_parametric:
        sampling_values = [rng.random(10), rng.random(10)]
        samples = rng.random((10, 10, p, m))
    else:
        sampling_values = rng.random(10)
        samples = rng.random((10, p, m))
    paaa = PAAAReductor(sampling_values, samples)
    rom = paaa.reduce(tol=1e-3)
    if is_parametric:
        assert rom.eval_tf(0, mu=0).shape == (p, m)
    else:
        assert rom.eval_tf(0).shape == (p, m)


def test_full_nd_loewner_row_mask():
    sampling_values = [np.array([0., 1., 3.]), np.array([10., 12., 15., 19.])]
    samples = np.arange(12).reshape(3, 4)**2
    interpolation_indices = [[0, 2], [1, 3]]
    interpolation_sets = [set(indices) for indices in interpolation_indices]

    expected = []
    for row in np.ndindex(samples.shape):
        if all(index in indices for index, indices in zip(row, interpolation_sets, strict=True)):
            continue
        entries = []
        for column in itertools.product(*interpolation_indices):
            factor = 1
            for dimension, (row_index, column_index) in enumerate(zip(row, column, strict=True)):
                if row_index in interpolation_sets[dimension]:
                    if row_index != column_index:
                        factor = 0
                        break
                else:
                    factor /= (sampling_values[dimension][row_index]
                               - sampling_values[dimension][column_index])
            entries.append((samples[row] - samples[column]) * factor)
        expected.append(entries)

    L = full_nd_loewner(samples, sampling_values, interpolation_indices)
    assert np.allclose(L, expected)

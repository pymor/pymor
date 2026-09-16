# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.transfer_function import TransferFunction
from pymor.reductors.aaa import PAAAReductor

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
        assert rom.eval_tf(sampling_values[0][0], mu=0).shape == (p, m)
    else:
        assert rom.eval_tf(0).shape == (p, m)
        assert rom.eval_tf(sampling_values[0]).shape == (p, m)


def test_paaa_sampled_named_parameters():
    fom = TransferFunction(1, 1, lambda s, mu: np.array([[1 / (s + mu['a'][0])]]), parameters={'a': 1})
    grid = [np.array([1., 2., 3., 4.]), np.array([1., 2., 3.])]
    samples = PAAAReductor.generate_samples(grid, fom)
    rom = PAAAReductor(grid, samples, parameters=fom.parameters, conjugate=False).reduce(tol=1e-10)
    assert rom.parameters == fom.parameters
    for i, s in enumerate(grid[0]):
        for j, a in enumerate(grid[1]):
            assert np.allclose(rom.eval_tf(s, mu={'a': a}), samples[i, j])
    with pytest.raises(ValueError, match='number of parameter sampling axes'):
        PAAAReductor(grid, samples, parameters={'a': 2})

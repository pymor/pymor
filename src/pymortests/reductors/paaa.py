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


@pytest.mark.parametrize(('post_process', 'num_nodes'), [(False, [5, 5]), (True, [5, 4])])
def test_paaa_post_processing(post_process, num_nodes):
    """Reproduce the order reduction in CRBG23, Section 3.2.1 and Table 1."""
    def transfer_function(s, p):
        return 1 / (1 + 25 * (s + p)**2) + 0.5 / (1 + 25 * (s - 0.5)**2) + 0.1 / (p + 25)

    sampling_values = [np.linspace(-1, 1, 21), np.linspace(0, 1, 21)]
    samples = transfer_function(sampling_values[0][:, np.newaxis], sampling_values[1][np.newaxis, :])
    # Leave room for roundoff when detecting the two-dimensional null space.
    reductor = PAAAReductor(sampling_values, samples, post_process=post_process, nsp_tol=1e-12)
    rom = reductor.reduce(tol=1e-12)

    assert [len(indices) for indices in reductor.itpl_part] == num_nodes
    # Check both the sampling grid and its midpoints.
    for s in np.linspace(-1, 1, 41):
        for p in np.linspace(0, 1, 41):
            assert np.allclose(rom.eval_tf(s, mu=p), transfer_function(s, p), rtol=1e-10, atol=1e-12)


def test_paaa_sampled_named_parameters():
    fom = TransferFunction(1, 1, lambda s, mu: np.array([[1 / (s + mu['a'][0])]]), parameters={'a': 1})
    grid = [np.array([1., 2., 3., 4.]), np.array([1., 2., 3.])]
    samples = PAAAReductor.generate_samples(grid, fom)
    rom = PAAAReductor(grid, samples, parameters=fom.parameters, force_real=False).reduce(tol=1e-10)
    assert rom.parameters == fom.parameters
    for i, s in enumerate(grid[0]):
        for j, a in enumerate(grid[1]):
            assert np.allclose(rom.eval_tf(s, mu={'a': a}), samples[i, j])
    with pytest.raises(AssertionError, match='number of parameter sampling axes'):
        PAAAReductor(grid, samples, parameters={'a': 2})

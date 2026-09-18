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


@pytest.mark.parametrize('initial_partition', [False, True])
def test_paaa_conjugate_interpolation(initial_partition):
    def transfer_function(s):
        return 1 / (s + 1) + 0.5 / (s + 3)

    nodes = 1j * np.arange(1, 5)
    # Keep the paired interpolation nodes instead of removing redundant nodes in postprocessing.
    reductor = PAAAReductor(nodes, transfer_function(nodes), force_real=True, post_process=False)
    itpl_part = [[1, 5]] if initial_partition else None  # 2j and its appended conjugate -2j
    rom = reductor.reduce(itpl_part=itpl_part, tol=1e-12)

    if initial_partition:
        assert reductor.itpl_part[0][:2] == [1, 5]
    selected_nodes = reductor.sampling_values[0][reductor.itpl_part[0]]
    assert len(selected_nodes) == 4
    assert np.all(np.isin(selected_nodes.conj(), selected_nodes))
    for s in 1j * np.linspace(-4.5, 4.5, 19):
        assert np.allclose(rom.eval_tf(s), transfer_function(s), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize(('dim_input', 'dim_output'), [(1, 1), (2, 3)])
def test_paaa_sampled_named_parameters(complex, dim_input, dim_output):
    gain = np.arange(1, dim_input * dim_output + 1).reshape(dim_output, dim_input)
    fom = TransferFunction(dim_input, dim_output, lambda s, mu: gain / (s + mu['a'][0]), parameters={'a': 1})
    grid = [np.array([1., 2., 3., 4.]), np.array([1., 2., 3.])]
    if complex:
        grid[0] = grid[0]*1j
    samples = PAAAReductor.generate_samples(grid, fom)
    rom = PAAAReductor(grid, samples, parameters=fom.parameters, force_real=False).reduce(tol=1e-10)
    assert rom.parameters == fom.parameters
    for i, s in enumerate(grid[0]):
        for j, a in enumerate(grid[1]):
            value = rom.eval_tf(s, mu={'a': a})
            assert value.shape == (dim_output, dim_input)
            assert np.allclose(value, samples[i, j])
    with pytest.raises(AssertionError, match='number of parameter sampling axes'):
        PAAAReductor(grid, samples, parameters={'a': 2})

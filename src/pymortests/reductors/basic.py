# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.gram_schmidt import gram_schmidt_biorth
from pymor.models.iosys import SecondOrderModel
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import SOLTIPGReductor

pytestmark = pytest.mark.builtin


def _so_model_and_bases(rng):
    n = 8
    M = np.diag(1 + 0.1 * np.arange(n))
    K = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    fom = SecondOrderModel.from_matrices(M, 0.1 * K + 0.05 * np.eye(n), K, rng.standard_normal((n, 2)),
                                         rng.standard_normal((2, n)), Cv=rng.standard_normal((2, n)))
    V = fom.solution_space.from_numpy(rng.standard_normal((n, 3)))
    W = fom.solution_space.from_numpy(rng.standard_normal((n, 3)))
    gram_schmidt_biorth(V, W, product=fom.M, copy=False)
    return fom, W, V


def _assert_same_transfer_function(rom1, rom2):
    for s in [0.5, 1j, 2 + 1j]:
        assert np.allclose(rom1.transfer_function.eval_tf(s), rom2.transfer_function.eval_tf(s))


@pytest.mark.parametrize('dim', [3, 2])
def test_soltipg_M_biorthonormal(dim, rng):
    fom, W, V = _so_model_and_bases(rng)

    rom = SOLTIPGReductor(fom, W, V, M_biorthonormal=True).reduce(dim)

    assert isinstance(rom.M, IdentityOperator)
    assert rom.order == dim
    _assert_same_transfer_function(rom, SOLTIPGReductor(fom, W[:dim], V[:dim]).reduce())


def test_soltipg_subbasis(rng):
    fom, W, V = _so_model_and_bases(rng)

    rom = SOLTIPGReductor(fom, W, V).reduce(2)

    _assert_same_transfer_function(rom, SOLTIPGReductor(fom, W[:2], V[:2]).reduce())

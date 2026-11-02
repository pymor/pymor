# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.models.iosys import PHLTIModel
from pymor.reductors.ph.basic import PHLTIPGReductor

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('dim', [3, 2])
@pytest.mark.parametrize('pg_projection', ['ph_preserving', 'energy_stable'])
def test_phltipg_QTE_orthonormal(pg_projection, dim, rng):
    n = 6
    J = rng.standard_normal((n, n))
    R = rng.standard_normal((n, n))
    fom = PHLTIModel.from_matrices(J - J.T, R @ R.T + np.eye(n), rng.standard_normal((n, 2)),
                                   Q=np.diag(1 + 0.1 * np.arange(n)))
    V = fom.solution_space.from_numpy(rng.standard_normal((n, 3)))
    gram_schmidt(V, product=fom.Q.H @ fom.E, copy=False)

    rom = PHLTIPGReductor(fom, V, QTE_orthonormal=True, pg_projection=pg_projection).reduce(dim)
    rom_ref = PHLTIPGReductor(fom, V[:dim], pg_projection=pg_projection).reduce()

    assert rom.order == dim
    for s in [0.5, 1j, 2 + 1j]:
        assert np.allclose(rom.transfer_function.eval_tf(s), rom_ref.transfer_function.eval_tf(s))

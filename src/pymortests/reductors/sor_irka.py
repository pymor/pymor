# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.iosys import SecondOrderModel
from pymor.reductors.sor_irka import SORIRKAReductor

pytestmark = pytest.mark.builtin


def test_sor_irka_projections(rng):
    n = 8
    K = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    fom = SecondOrderModel.from_matrices(np.diag(1 + 0.1 * np.arange(n)), 0.1 * K + 0.05 * np.eye(n), K,
                                         rng.standard_normal((n, 2)), rng.standard_normal((2, n)))

    roms = [SORIRKAReductor(fom).reduce(2, maxit=3, projection=projection) for projection in ('orth', 'biorth')]

    for rom in roms:
        assert isinstance(rom, SecondOrderModel)
        assert rom.order == 2
    for s in [0.5, 1j, 2 + 1j]:
        assert np.allclose(roms[0].transfer_function.eval_tf(s), roms[1].transfer_function.eval_tf(s))

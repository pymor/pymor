# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.sparse as sps

from pymor.models.iosys import LTIModel
from pymor.reductors.mt import MTReductor

pytestmark = pytest.mark.builtin


test_data = [
    (8, {}),
    (8, {'which': 'NS'}),
    (8, {'which': 'NM'}),
    (8, {'projection': 'biorth'}),
    (8, {'method_options': {'tol': 1e-12}}),
    (2, {'decomposition': 'eig'}),
    (2, {'decomposition': 'eig', 'which': 'NS'}),
    (2, {'decomposition': 'eig', 'which': 'NM'}),
    (2, {'decomposition': 'eig', 'which': 'SM'}),
    (2, {'decomposition': 'eig', 'which': 'LR'}),
    (2, {'decomposition': 'eig', 'symmetric': True}),
    (3, {'decomposition': 'eig'}),  # dominant poles not closed under conjugation
    (3, {'decomposition': 'eig', 'allow_complex_rom': True}),
]


@pytest.mark.parametrize('r,mt_kwargs', test_data)
def test_mt(r, mt_kwargs):
    n = 10
    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.ones((n, 1))
    B[:6] = 10
    C = B.T
    fom = LTIModel.from_matrices(A, B, C)
    mt = MTReductor(fom)

    rom = mt.reduce(r, **mt_kwargs)
    assert isinstance(rom, LTIModel) and rom.order == r

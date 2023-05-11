# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest

from pymor.models.iosys import LTIModel
from pymor.reductors.mt import MTReductor
from pymortests.reductors.loewner import make_fom

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
    fom = make_fom(10)
    mt = MTReductor(fom)

    rom = mt.reduce(r, **mt_kwargs)
    assert isinstance(rom, LTIModel) and rom.order == r

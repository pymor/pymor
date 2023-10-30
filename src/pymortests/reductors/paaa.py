# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

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


@pytest.mark.parametrize('m,p,is_parametric', test_data)
def test_paaa(m,p,is_parametric):
    np.random.seed(0)
    if is_parametric:
        sampling_values = [np.random.rand(10), np.random.rand(10)]
        samples = np.random.rand(10, 10, p, m)
    else:
        sampling_values = np.random.rand(10)
        samples = np.random.rand(10, p, m)
    paaa = PAAAReductor(sampling_values, samples)
    rom = paaa.reduce(tol=1e-3)
    if is_parametric:
        assert rom.eval_tf(0, mu=0).shape == (p, m)
    else:
        assert rom.eval_tf(0).shape == (p, m)

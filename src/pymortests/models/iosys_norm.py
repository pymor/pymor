# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest

from pymortests.models.iosys_add_sub_mul import get_model

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('parametric', [False, True])
def test_hinf_norm(parametric):
    m = get_model('LTIModel', 0, parametric)

    if parametric:
        mu = m.parameters.parse([-1.])
    else:
        mu = None

    hinf_norm = m.hinf_norm(mu=mu)
    hinf_norm_f, fpeak = m.hinf_norm(mu=mu, return_fpeak=True)

    assert hinf_norm == hinf_norm_f

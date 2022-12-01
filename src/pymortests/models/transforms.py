# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from itertools import combinations

from pymor.models.iosys import LTIModel
from pymor.models.transforms import BilinearTransformation, CayleyTransformation, MoebiusTransformation


type_list = ['BilinearTransformation', 'CayleyTransformation', 'MoebiusTransformation']
points_list = list(combinations([0, 1, -1, 1j, -1j, np.inf], 3))
sampling_time_list = [0, 1/100, 2]


def get_transformation(name):
    if name == 'BilinearTransformation':
        return BilinearTransformation(2)
    elif name == 'CayleyTransformation':
        return CayleyTransformation()
    elif name == 'MoebiusTransformation':
        return MoebiusTransformation([1+2j, 2+3j, 4+5j, 6+7j])
    else:
        raise KeyError


@pytest.mark.parametrize('m', type_list)
def test_inv(m):
    m = get_transformation(m)
    m_inv = m.inverse()
    mm_inv = MoebiusTransformation((m @ m_inv).coefficients, normalize=True)
    m_invm = MoebiusTransformation((m_inv @ m).coefficients, normalize=True)
    assert np.allclose(np.eye(2).ravel(), mm_inv.coefficients)
    assert np.allclose(np.eye(2).ravel(), m_invm.coefficients)


@pytest.mark.parametrize('m1', type_list)
@pytest.mark.parametrize('m2', type_list)
def test_matmul(m1, m2):
    m1 = get_transformation(m1)
    m2 = get_transformation(m2)
    m = m1 @ m2
    assert np.allclose(m.coefficients.reshape(2, 2), m1.coefficients.reshape(2, 2) @ m2.coefficients.reshape(2, 2))


@pytest.mark.parametrize('p1', points_list)
@pytest.mark.parametrize('p2', points_list)
def test_from_points(p1, p2):
    p1, p2 = np.asarray(p1), np.asarray(p2)
    M1 = MoebiusTransformation.from_points(p1, z=p2)
    m1 = M1(p2)

    for i in range(3):
        if np.isinf(p1[i]):
            assert np.isinf(m1[i]) or np.abs(m1[i]) >= 1e+15 or np.allclose(p1[i], m1[i])
        elif np.isinf(m1[i]):
            assert np.abs(p1[i]) >= 1e+15 or np.allclose(p1[i], m1[i])
        else:
            assert np.allclose(p1[i], m1[i])


@pytest.mark.parametrize('sampling_time', sampling_time_list)
def test_substitution(sampling_time):
    sys1 = LTIModel.from_matrices(np.array([-1]), np.array([[1]]), np.array([[1]]), sampling_time=sampling_time)
    if sampling_time:
        sys2 = sys1.to_continuous()
        assert isinstance(sys2, LTIModel)
        assert sys2.sampling_time == 0
    else:
        sys2 = sys1.to_discrete(1)
        assert isinstance(sys2, LTIModel)
        assert sys2.sampling_time == 1

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import combinations

import numpy as np
import pytest

from pymor.models.iosys import LTIModel
from pymor.models.transforms import BilinearTransformation, CayleyTransformation, MoebiusTransformation

type_list = ['BilinearTransformation', 'CayleyTransformation', 'MoebiusTransformation-real',
             'MoebiusTransformation-complex']
points_list = list(combinations([0, 1, -1, 1j, -1j, np.inf], 3))
sampling_time_list = [0, 1/100, 2]

pytestmark = pytest.mark.builtin


def get_transformation(name):
    if name == 'BilinearTransformation':
        return BilinearTransformation(2)
    elif name == 'CayleyTransformation':
        return CayleyTransformation()
    elif name == 'MoebiusTransformation-real':
        return MoebiusTransformation([1, 2, 3, 4])
    elif name == 'MoebiusTransformation-complex':
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


@pytest.mark.parametrize('m1', type_list)
def test_normalization(m1):
    m1 = get_transformation(m1)
    m2 = MoebiusTransformation(m1.coefficients, normalize=True)
    a, b, c, d = m2.coefficients
    assert np.isrealobj(m1.coefficients) == np.isrealobj(m2.coefficients)
    assert np.allclose(m1(0), m2(0))
    assert np.allclose(m1(1), m2(1))
    assert np.allclose(m1(np.inf), m2(np.inf))

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
def test_tustin(sampling_time):
    sys1 = LTIModel.from_matrices(np.array([1]), np.array([[1]]), np.array([[1]]), sampling_time=sampling_time)
    if sampling_time:
        sys2 = sys1.to_continuous()
        assert isinstance(sys2, LTIModel)
        assert sys2.sampling_time == 0
        A2, B2, C2, D2, E2 = sys2.to_discrete(sampling_time).to_matrices()
    else:
        sys2 = sys1.to_discrete(1)
        assert isinstance(sys2, LTIModel)
        assert sys2.sampling_time == 1
        A2, B2, C2, D2, E2 = sys2.to_continuous().to_matrices()

    D2 = np.zeros((C2.shape[0], B2.shape[1])) if D2 is None else D2
    E2 = np.eye(A2.shape[0]) if E2 is None else E2
    A1, B1, C1, D1, E1 = sys1.to_matrices()
    D1 = np.zeros((C1.shape[0], B1.shape[1])) if D1 is None else D1
    E1 = np.eye(A1.shape[0]) if E1 is None else E1
    assert np.allclose(A1, A2)
    assert np.allclose(B1, B2)
    assert np.allclose(C1, C2)
    assert np.allclose(D1, D2)
    assert np.allclose(E1, E2)

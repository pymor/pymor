# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.iosys import LTIModel
from pymor.models.pole_residue import PoleResidueTransferFunction
from pymor.models.transfer_function import TransferFunction
from pymor.operators.constructions import ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymortests.core.pickling import assert_picklable
from pymortests.models.iosys_add_sub_mul import (
    assert_tf_add,
    assert_tf_mul,
    assert_tf_sub,
    get_model,
    get_tf,
    name_list,
)

pytestmark = pytest.mark.builtin


sampling_time_list = [0, 1]
s_list = [0.5, 1j, 2 + 3j]


def get_lti(rng, m, p, with_D, with_E, sampling_time, n=6):
    A = rng.standard_normal((n, n)) - n * np.eye(n)
    B = rng.standard_normal((n, m))
    C = rng.standard_normal((p, n))
    D = rng.standard_normal((p, m)) if with_D else None
    E = rng.standard_normal((n, n)) + n * np.eye(n) if with_E else None
    return LTIModel.from_matrices(A, B, C, D, E, sampling_time=sampling_time)


def get_pole_residue_tf(rng, m, p, with_D, sampling_time):
    poles = np.array([-1, -2 + 3j, -2 - 3j, -0.5 + 1j, -0.5 - 1j])
    b = rng.standard_normal((3, m)) + 1j * rng.standard_normal((3, m))
    c = rng.standard_normal((3, p)) + 1j * rng.standard_normal((3, p))
    b[0] = b[0].real
    c[0] = c[0].real
    b = np.vstack([b[0], b[1], b[1].conj(), b[2], b[2].conj()])
    c = np.vstack([c[0], c[1], c[1].conj(), c[2], c[2].conj()])
    D = rng.standard_normal((p, m)) if with_D else None
    return PoleResidueTransferFunction(poles, b, c, D, sampling_time=sampling_time)


def assert_same_tf(m1, m2, mu=None):
    tf1 = get_tf(m1)
    tf2 = get_tf(m2)
    for s in s_list:
        assert np.allclose(tf1.eval_tf(s, mu=mu), tf2.eval_tf(s, mu=mu))
        assert np.allclose(tf1.eval_dtf(s, mu=mu), tf2.eval_dtf(s, mu=mu))


@pytest.mark.parametrize('sampling_time', sampling_time_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_D', [False, True])
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('m', [1, 3])
def test_to_pole_residue(rng, m, p, with_D, with_E, sampling_time):
    lti = get_lti(rng, m, p, with_D, with_E, sampling_time)
    pr = lti.to_pole_residue()
    assert type(pr) is PoleResidueTransferFunction
    assert pr.order == lti.order
    assert pr.dim_input == m
    assert pr.dim_output == p
    assert pr.sampling_time == sampling_time
    assert pr.poles.shape == (lti.order,)
    assert pr.b.shape == (lti.order, m)
    assert pr.c.shape == (lti.order, p)
    assert (pr.D is None) == (not with_D)
    assert_same_tf(pr, lti)


def test_to_pole_residue_parametric(rng):
    n = 4
    A = (NumpyMatrixOperator(rng.standard_normal((n, n)) - n * np.eye(n))
         + ProjectionParameterFunctional('mu') * NumpyMatrixOperator(np.eye(n)))
    B = NumpyMatrixOperator(rng.standard_normal((n, 1)))
    C = NumpyMatrixOperator(rng.standard_normal((1, n)))
    lti = LTIModel(A, B, C)
    for mu in (-1, 1):
        pr = lti.to_pole_residue(mu=mu)
        assert not pr.parametric
        assert np.allclose(np.sort_complex(pr.poles), np.sort_complex(lti.poles(mu=mu)))
        for s in s_list:
            assert np.allclose(pr.eval_tf(s), lti.transfer_function.eval_tf(s, mu=mu))


@pytest.mark.parametrize('sampling_time', sampling_time_list)
@pytest.mark.parametrize('with_D', [False, True])
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('m', [1, 3])
def test_to_lti(rng, m, p, with_D, sampling_time):
    pr = get_pole_residue_tf(rng, m, p, with_D, sampling_time)
    lti = pr.to_lti()
    assert type(lti) is LTIModel
    assert lti.order == pr.order
    assert lti.sampling_time == sampling_time
    assert all(np.isrealobj(M) for M in lti.to_abcde_matrices() if M is not None)
    assert isinstance(lti.D, ZeroOperator) == (not with_D)
    assert_same_tf(lti, pr)


def test_to_lti_explicit():
    lti = PoleResidueTransferFunction(np.array([-1, -2 + 1j, -2 - 1j]),
                                      np.array([[1], [1 + 1j], [1 - 1j]]),
                                      np.array([[2], [3j], [-3j]])).to_lti()
    for s in s_list:
        tf = 2 / (s + 1) + 3j * (1 + 1j) / (s + 2 - 1j) - 3j * (1 - 1j) / (s + 2 + 1j)
        assert np.allclose(lti.transfer_function.eval_tf(s), [[tf]])


@pytest.mark.parametrize('with_D', [False, True])
def test_lti_round_trip(rng, with_D):
    lti = get_lti(rng, 2, 2, with_D, False, 0)
    lti2 = lti.to_pole_residue().to_lti()
    assert lti2.order == lti.order
    assert np.allclose(np.sort_complex(lti2.poles()), np.sort_complex(lti.poles()))
    assert_same_tf(lti2, lti)


@pytest.mark.parametrize('with_D2', [False, True])
@pytest.mark.parametrize('with_D1', [False, True])
def test_add_sub_neg(rng, with_D1, with_D2):
    pr1 = get_pole_residue_tf(rng, 2, 3, with_D1, 0)
    pr2 = get_pole_residue_tf(rng, 2, 3, with_D2, 0)
    pr_add = pr1 + pr2
    pr_sub = pr1 - pr2
    pr_neg = -pr1
    assert type(pr_add) is type(pr_sub) is type(pr_neg) is PoleResidueTransferFunction
    assert pr_add.order == pr_sub.order == pr1.order + pr2.order
    assert (pr_add.D is None) == (pr_sub.D is None) == (not with_D1 and not with_D2)
    assert (pr_neg.D is None) == (not with_D1)
    for s in s_list:
        assert np.allclose(pr_add.eval_tf(s), pr1.eval_tf(s) + pr2.eval_tf(s))
        assert np.allclose(pr_add.eval_dtf(s), pr1.eval_dtf(s) + pr2.eval_dtf(s))
        assert np.allclose(pr_sub.eval_tf(s), pr1.eval_tf(s) - pr2.eval_tf(s))
        assert np.allclose(pr_sub.eval_dtf(s), pr1.eval_dtf(s) - pr2.eval_dtf(s))
        assert np.allclose(pr_neg.eval_tf(s), -pr1.eval_tf(s))
        assert np.allclose(pr_neg.eval_dtf(s), -pr1.eval_dtf(s))


@pytest.mark.parametrize('parametric', [False, True])
@pytest.mark.parametrize('sampling_time', sampling_time_list)
@pytest.mark.parametrize('pr_first', [False, True])
@pytest.mark.parametrize('name', name_list)
def test_mixed_arithmetic(rng, name, pr_first, sampling_time, parametric):
    pr = get_pole_residue_tf(rng, 1, 1, True, sampling_time)
    other = get_model(name, sampling_time, parametric)
    m1, m2 = (pr, other) if pr_first else (other, pr)
    for m, assert_tf in ((m1 + m2, assert_tf_add), (m1 - m2, assert_tf_sub), (m1 * m2, assert_tf_mul)):
        assert type(m) is TransferFunction
        assert_tf(m1, m2, m, parametric)


def test_arrays_read_only():
    poles = np.array([-1., -2.])
    b = np.ones((2, 1))
    pr = PoleResidueTransferFunction(poles, b, np.ones((2, 1)))
    assert poles.flags.writeable
    assert b.flags.writeable
    assert not pr.poles.flags.writeable
    assert not pr.b.flags.writeable
    assert not pr.c.flags.writeable


def test_pickle(rng):
    assert_picklable(get_pole_residue_tf(rng, 2, 1, True, 0))

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Cross-representation checks, enabled when the pole-residue branch is present."""

import numpy as np
import pytest

from pymor.models.barycentric import BarycentricRationalTransferFunction
from pymor.models.transfer_function import TransferFunction

PoleResidueTransferFunction = pytest.importorskip('pymor.models.pole_residue').PoleResidueTransferFunction

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('sampling_time', [0, 0.1])
@pytest.mark.parametrize('barycentric_first', [False, True])
def test_mixed_arithmetic(sampling_time, barycentric_first):
    values = np.stack([np.ones((2, 2)) / 3, np.ones((2, 2)) / 4])
    bary = BarycentricRationalTransferFunction([0, 1], values, [-3, 4], sampling_time=sampling_time)
    pole = PoleResidueTransferFunction([-1, -2], [[1, 2], [2, 1]], [[2, 1], [1, 3]],
                                      D=np.eye(2), sampling_time=sampling_time)
    a, b = (bary, pole) if barycentric_first else (pole, bary)
    for operation in ('add', 'sub', 'mul'):
        result = {'add': lambda: a + b, 'sub': lambda: a - b, 'mul': lambda: a * b}[operation]()
        assert type(result) is TransferFunction
        assert result.sampling_time == sampling_time
        for s in [0, 1, 0.3j]:
            av, bv = a.eval_tf(s), b.eval_tf(s)
            ad, bd = a.eval_dtf(s), b.eval_dtf(s)
            if operation == 'mul':
                expected, derivative = av @ bv, ad @ bv + av @ bd
            elif operation == 'add':
                expected, derivative = av + bv, ad + bd
            else:
                expected, derivative = av - bv, ad - bd
            np.testing.assert_allclose(result.eval_tf(s), expected)
            np.testing.assert_allclose(result.eval_dtf(s), derivative)


@pytest.mark.parametrize('barycentric_first', [False, True])
def test_rectangular_mixed_composition(barycentric_first):
    values = np.stack([np.ones((3, 2)) / 3, np.ones((3, 2)) / 4])
    bary = BarycentricRationalTransferFunction([0, 1], values, [-3, 4])
    p, m = (2, 4) if barycentric_first else (4, 3)
    pole = PoleResidueTransferFunction([-2], np.ones((1, m)), np.ones((1, p)))
    a, b = (bary, pole) if barycentric_first else (pole, bary)
    result = a * b
    assert (result.dim_output, result.dim_input) == (a.dim_output, b.dim_input)
    np.testing.assert_allclose(result.eval_tf(1j), a.eval_tf(1j) @ b.eval_tf(1j))
    np.testing.assert_allclose(result.eval_dtf(1j), a.eval_dtf(1j) @ b.eval_tf(1j)
                               + a.eval_tf(1j) @ b.eval_dtf(1j))


def test_mixed_h2_inner_product():
    values = np.stack([np.ones((2, 2)) / 3, np.ones((2, 2)) / 4])
    bary = BarycentricRationalTransferFunction([0, 1], values, [-3, 4])
    pole = PoleResidueTransferFunction([-2], [[1, 2]], [[2, 1]])
    assert np.isclose(bary.h2_inner(pole), 9 / 5)
    assert np.isclose(pole.h2_inner(bary), 9 / 5)

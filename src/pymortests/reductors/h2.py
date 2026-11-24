# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.h2 import (
    GapIRKAReductor,
    IRKAReductor,
    OneSidedIRKAReductor,
    TFIRKAReductor,
    TSIAReductor,
    VectorFittingReductor,
)

pytestmark = pytest.mark.builtin

IRKA_TYPE_REDUCTORS = {
    'irka': (IRKAReductor, {}),
    'one_sided_V': (OneSidedIRKAReductor, {'version': 'V'}),
    'one_sided_W': (OneSidedIRKAReductor, {'version': 'W'}),
    'tsia': (TSIAReductor, {}),
    'tf_irka': (TFIRKAReductor, {}),
    'gap_irka': (GapIRKAReductor, {}),
}


def _random_mimo_lti(rng):
    n, m, p = 10, 2, 2
    A = -np.diag(np.arange(1, n + 1)) + 0.5 * rng.standard_normal((n, n))
    return LTIModel.from_matrices(A, rng.standard_normal((n, m)), rng.standard_normal((p, n)))


def test_vector_fitting_sampled_transfer_function():
    fom = TransferFunction(1, 1, lambda s: np.array([[1 / (s + 1)]]))

    s = 1j * np.arange(1, 3)
    Hs = VectorFittingReductor.generate_samples(s, fom)
    reductor = VectorFittingReductor(s, Hs, weights=np.array([2., 3.]))

    assert np.array_equal(reductor.s, [1j, 2j, -1j, -2j])
    assert np.allclose(reductor.Hs, [1 / (1 + 1j), 1 / (1 + 2j), 1 / (1 - 1j), 1 / (1 - 2j)])
    assert np.allclose(reductor.weights_sqrt**2, [2., 3., 2., 3.])


def test_irka():
    A = np.array([[-1, 0], [0, -2]])
    B = np.array([[1], [2]])
    C = np.array([[2, 1]])
    fom = LTIModel.from_matrices(A, B, C)
    irka = IRKAReductor(fom)

    rom = irka.reduce(1)
    assert isinstance(rom, LTIModel)
    assert rom.order == 1

    rom = irka.reduce(np.array([1]))
    assert isinstance(rom, LTIModel)
    assert rom.order == 1

    rom = irka.reduce({'sigma': np.array([1]),
                       'b': np.array([[1]]),
                       'c': np.array([[1]])})
    assert isinstance(rom, LTIModel)
    assert rom.order == 1

    rom = irka.reduce(LTIModel.from_matrices(np.array([[-1]]),
                                             np.array([[1]]),
                                             np.array([[1]])))
    assert isinstance(rom, LTIModel)
    assert rom.order == 1


def test_tfirka_numerical_conjugate_pairs():
    blocks = [np.array([[-damping, frequency], [-frequency, -damping]])
              for damping, frequency in [(0.05, 2), (0.1, 5), (0.2, 9)]]
    A = np.block([[blocks[i] if i == j else np.zeros((2, 2)) for j in range(3)] for i in range(3)])
    B = np.array([[1], [0.3], [0.5], [1.2], [-0.4], [0.8]])
    C = np.array([[0.7, -0.2, 1, 0.4, 0.2, -0.8]])
    fom = LTIModel.from_matrices(A, B, C)
    sigma = np.array([0.1 + 2j, 0.1 - 2j, 0.2 + 5j, 0.2 - 5j])
    initial_data = {'sigma': sigma, 'b': np.ones((4, 1)), 'c': np.ones((4, 1))}

    rom = TFIRKAReductor(fom).reduce(initial_data, maxit=2, tol=1e-12)

    assert all(np.isrealobj(to_matrix(getattr(rom, operator), format='dense')) for operator in 'ABCE')


@pytest.mark.parametrize('kwargs', [{'conv_crit': 'h2'}, {'tol': -1}, {'maxit': 0}, {'num_prev': 0}])
def test_gap_irka_invalid_args(kwargs, rng):
    with pytest.raises(AssertionError):
        GapIRKAReductor(_random_mimo_lti(rng)).reduce(2, **kwargs)


@pytest.mark.parametrize('conv_crit', ['sigma', 'htwogap', 'ltwo'])
def test_gap_irka_conv_crit_independent_of_projection(conv_crit, rng):
    fom = _random_mimo_lti(rng)
    conv_crits = []
    for projection in ('orth', 'biorth'):
        gap_irka = GapIRKAReductor(fom)
        gap_irka.reduce(2, tol=1e-14, maxit=4, conv_crit=conv_crit, projection=projection)
        conv_crits.append(gap_irka.conv_crit)

    assert np.allclose(*conv_crits)


@pytest.mark.parametrize('rom0_kind', ['order', 'sigma', 'dict', 'rom'])
@pytest.mark.parametrize('name', IRKA_TYPE_REDUCTORS)
def test_irka_type_reductors(name, rom0_kind, rng):
    fom = _random_mimo_lti(rng)
    rom0_params = {
        'order': 2,
        'sigma': np.array([1., 2.]),
        'dict': {'sigma': np.array([1., 2.]), 'b': np.ones((2, 2)), 'c': np.ones((2, 2))},
        'rom': LTIModel.from_matrices(-np.diag([1., 2.]), np.ones((2, 2)), np.ones((2, 2))),
    }[rom0_kind]
    cls, kwargs = IRKA_TYPE_REDUCTORS[name]
    reductor = cls(fom, **kwargs)

    rom = reductor.reduce(rom0_params, tol=1e-14, maxit=3)

    assert isinstance(rom, LTIModel)
    assert rom.order == 2
    assert len(reductor.conv_crit) == 3
    assert len(reductor.sigma_list) == 4
    if cls is TFIRKAReductor:
        assert reductor.V is None
        assert reductor.W is None
    else:
        assert len(reductor.V) == len(reductor.W) == 2
        assert reductor.reconstruct(rom.solution_space.ones()) in fom.solution_space


def test_irka_parametric_compute_errors():
    n = 10
    p = ProjectionParameterFunctional('p')
    A = NumpyMatrixOperator(-np.diag(np.arange(1, n + 1))) + p * NumpyMatrixOperator(-np.eye(n))
    fom = LTIModel(A, NumpyMatrixOperator(np.ones((n, 1))), NumpyMatrixOperator(np.ones((1, n))))
    irka = IRKAReductor(fom, mu=fom.parameters.parse(1))

    irka.reduce(2, maxit=2, compute_errors=True)

    assert len(irka.errors) == len(irka.conv_crit)
    assert all(0 < error < 1 for error in irka.errors)

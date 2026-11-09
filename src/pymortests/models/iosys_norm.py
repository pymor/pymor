# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.sparse as sps

from pymor.bindings.scipy import ScipyLyapunovSolver
from pymor.models.iosys import LTIModel
from pymor.operators.numpy import NumpyGenericOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.solvers.matrix_equations.default import MatrixEquationSolvers
from pymortests.models.iosys_add_sub_mul import get_model

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('typ', [
    'c_lr', 'o_lr', 'c_dense', 'o_dense', 'bs_c_lr', 'bs_o_lr', 'lqg_c_lr', 'lqg_o_lr',
    'pr_c_lr', 'pr_o_lr', 'pr_c_dense', 'pr_o_dense', ('br_c_lr', 2), ('br_o_lr', 2),
])
@pytest.mark.parametrize('complex_part', ['A', 'E'])
@pytest.mark.parametrize('matrix_type', [np.asarray, sps.csr_matrix])
def test_gramian_complex_state_operators_not_supported(typ, complex_part, matrix_type):
    matrices = {'A': -2 * np.eye(1), 'B': np.eye(1), 'C': np.eye(1), 'D': np.eye(1), 'E': np.eye(1)}
    matrices[complex_part] = matrix_type((1 + 1j) * matrices[complex_part])
    model = LTIModel.from_matrices(**matrices)

    with pytest.raises(NotImplementedError, match=f'complex {complex_part}'):
        model.gramian(typ)


@pytest.mark.parametrize(('typ', 'complex_part'), [
    ('c_dense', 'B'), ('o_dense', 'C'), ('lqg_c_lr', 'C'), ('lqg_o_lr', 'B'),
    ('pr_c_dense', 'D'), ('pr_o_dense', 'D'), (('br_c_lr', 2), 'B'), (('br_o_lr', 2), 'C'),
])
def test_gramian_complex_input_output_data_not_supported(typ, complex_part):
    matrices = {'A': -2 * np.eye(1), 'B': np.eye(1), 'C': np.eye(1), 'D': np.eye(1)}
    matrices[complex_part] = (1 + 1j) * matrices[complex_part]
    model = LTIModel.from_matrices(**matrices)

    with pytest.raises(NotImplementedError, match=f'complex {complex_part}'):
        model.gramian(typ)


@pytest.mark.parametrize('mu', [1., 1. + 1j])
def test_gramian_evaluates_parameter_realness(mu):
    model = LTIModel.from_matrices(-2 * np.eye(1), np.eye(1), np.eye(1))
    model = model.with_(A=model.A * ProjectionParameterFunctional('a'))

    if np.iscomplexobj(mu):
        with pytest.raises(NotImplementedError, match='complex A'):
            model.gramian('c_dense', mu={'a': mu})
    else:
        np.testing.assert_allclose(model.gramian('c_dense', mu={'a': mu}), [[0.25]])


@pytest.mark.parametrize(('typ', 'complex_part'), [('c_dense', 'C'), ('o_dense', 'B')])
def test_gramian_ignores_unused_complex_operators(typ, complex_part):
    matrices = {'A': -2 * np.eye(1), 'B': np.eye(1), 'C': np.eye(1), 'D': 1j * np.eye(1)}
    matrices[complex_part] = 1j * matrices[complex_part]
    model = LTIModel.from_matrices(**matrices)

    np.testing.assert_allclose(model.gramian(typ), [[0.25]])


@pytest.mark.parametrize(('dim_input', 'dim_output', 'complex_part'), [(1, 2, 'B'), (2, 1, 'C')])
def test_h2_norm_rejects_unsupported_complex_gramian_data(dim_input, dim_output, complex_part):
    matrices = {'A': -2 * np.eye(1), 'B': np.ones((1, dim_input)), 'C': np.ones((dim_output, 1))}
    matrices[complex_part] = 1j * matrices[complex_part]
    model = LTIModel.from_matrices(**matrices)

    with pytest.raises(NotImplementedError, match=f'complex {complex_part}'):
        model.h2_norm()


def test_h2_norm_can_use_real_gramian_with_complex_output():
    model = LTIModel.from_matrices(-2 * np.eye(1), np.eye(1), 1j * np.eye(1))
    np.testing.assert_allclose(model.h2_norm(), 0.5)


def test_gramian_does_not_materialize_opaque_operators(monkeypatch):
    def mapping(U, mu=None):
        pytest.fail('Opaque operator was applied during validation')

    A = NumpyGenericOperator(mapping, adjoint_mapping=mapping, linear=True, dim_source=1, dim_range=1)
    model = LTIModel.from_matrices(-2 * np.eye(1), np.eye(1), np.eye(1)).with_(
        A=A, matrix_equation_solvers=MatrixEquationSolvers(lyapunov=ScipyLyapunovSolver()),
    )

    def solve(solver, equation):
        assert equation.A is A
        return np.eye(1)

    monkeypatch.setattr(ScipyLyapunovSolver, '_solve', solve)
    np.testing.assert_allclose(model.gramian('c_dense'), np.eye(1))


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

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain, product

import numpy as np
import pytest
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.solvers.matrix_equations.equations import PositiveRiccatiEquation, RiccatiEquation
from pymor.solvers.matrix_equations.utils import chol
from pymortests.solvers.matrix_equations.lyapunov import (
    conv_diff_1d_fd,
    conv_diff_1d_fem,
    fro_norm,
    skip_if_missing_solver,
)

pytestmark = pytest.mark.builtin


n_list_small = [10, 20]
n_list_big = [250]
m_list = [1, 2]
p_list = [1, 2]
ricc_lr_backend_list_small = [
    'scipy',
    'slycot',
]
ricc_lr_backend_list_big = [
    'radi'
]
ricc_dense_backend_list = [
    'scipy',
    'slycot'
]


def relative_residual(A, E, B, C, R, Q, S, Z, trans, positive=False):
    if not trans:
        if E is None:
            linear = A @ Z @ Z.T
            quadratic = Z
        else:
            linear = A @ Z @ (Z.T @ E.T)
            quadratic = E @ Z
        quadratic = quadratic @ (Z.T @ C.T)
        if S is not None:
            quadratic = quadratic + S.T
        RHS = B @ B.T if R is None else B @ R @ B.T
        quadratic_weight = Q
    else:
        if E is None:
            linear = A.T @ Z @ Z.T
            quadratic = Z
        else:
            linear = A.T @ Z @ (Z.T @ E)
            quadratic = E.T @ Z
        quadratic = quadratic @ (Z.T @ B)
        if S is not None:
            quadratic = quadratic + S
        RHS = C.T @ C if Q is None else C.T @ Q @ C
        quadratic_weight = R
    linear += linear.T
    if quadratic_weight is None:
        quadratic = quadratic @ quadratic.T
    else:
        quadratic = quadratic @ spla.solve(quadratic_weight, quadratic.T)
    res = fro_norm(linear + quadratic + RHS if positive else linear - quadratic + RHS)
    rhs = fro_norm(RHS)
    return res / rhs


@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('with_Q', [False, True])
@pytest.mark.parametrize('with_S', [False, True])
@pytest.mark.parametrize('trans',  [False, True])
@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('backend', ricc_dense_backend_list)
def test_ricc_dense(n, m, p, with_E, with_R, with_Q, with_S, trans, backend, rng):
    skip_if_missing_solver(backend)

    mat_old = []
    mat_new = []
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        A = A.toarray()
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        A = A.toarray()
        E = E.toarray()
        mat_old.append(E.copy())
        mat_new.append(E)
    A = np.asfortranarray(A)
    mat_old.append(A.copy())
    mat_new.append(A)
    B = rng.standard_normal((n, m))
    mat_old.append(B.copy())
    mat_new.append(B)
    C = rng.standard_normal((p, n))
    mat_old.append(C.copy())
    mat_new.append(C)
    D = rng.standard_normal((p, m))
    R0 = rng.standard_normal((m, m))
    Q0 = rng.standard_normal((p, p))
    R = D.T.dot(D) + R0.dot(R0.T) if with_R else None
    Q = D.dot(D.T) + Q0.dot(Q0.T) if with_Q else None
    if not trans:
        S = 1e-1 * D @ B.T if with_S else None
    else:
        S = 1e-1 * C.T @ D if with_S else None
    if with_R:
        mat_old.append(R.copy())
        mat_new.append(R)
    if with_Q:
        mat_old.append(Q.copy())
        mat_new.append(Q)
    if with_S:
        mat_old.append(S.copy())
        mat_new.append(S)

    equation = RiccatiEquation.from_matrices(A, E, B, C, R, Q, S, trans=trans)


    if backend == 'slycot':
        from pymor.bindings.slycot import SlycotRiccatiSolver
        solver = SlycotRiccatiSolver()
    elif backend == 'scipy':
        from pymor.bindings.scipy import ScipyRiccatiSolver
        solver = ScipyRiccatiSolver()
    else:
        raise ValueError

    X = equation.solve(solver=solver)

    assert relative_residual(A, E, B, C, R, Q, S, chol(X), trans) < 1e-8

    for mat1, mat2 in zip(mat_old, mat_new, strict=True):
        assert type(mat1) is type(mat2)
        assert np.all(mat1 == mat2)


@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('with_Q', [False, True])
@pytest.mark.parametrize('with_S', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('backend', ricc_dense_backend_list)
def test_pos_ricc_dense(n, m, p, with_E, with_R, with_Q, with_S, trans, backend, rng):
    skip_if_missing_solver(backend)

    mat_old = []
    mat_new = []
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        A = A.toarray()
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        A = A.toarray()
        E = E.toarray()
        mat_old.append(E.copy())
        mat_new.append(E)
    A = np.asfortranarray(A)
    mat_old.append(A.copy())
    mat_new.append(A)
    B = rng.standard_normal((n, m))
    mat_old.append(B.copy())
    mat_new.append(B)
    C = rng.standard_normal((p, n))
    mat_old.append(C.copy())
    mat_new.append(C)
    D = rng.standard_normal((p, m))
    R0 = rng.standard_normal((m, m))
    Q0 = rng.standard_normal((p, p))
    R = D.T.dot(D) + R0.dot(R0.T) if with_R else None
    Q = D.dot(D.T) + Q0.dot(Q0.T) if with_Q else None
    if not trans:
        S = 1e-1 * D @ B.T if with_S else None
    else:
        S = 1e-1 * C.T @ D if with_S else None
    if with_R:
        mat_old.append(R.copy())
        mat_new.append(R)
    if with_Q:
        mat_old.append(Q.copy())
        mat_new.append(Q)
    if with_S:
        mat_old.append(S.copy())
        mat_new.append(S)

    equation = PositiveRiccatiEquation.from_matrices(A, E, B, C, R, Q, S, trans=trans)

    if backend == 'slycot':
        from pymor.bindings.slycot import SlycotPositiveRiccatiSolver
        solver = SlycotPositiveRiccatiSolver()
    elif backend == 'scipy':
        from pymor.bindings.scipy import ScipyPositiveRiccatiSolver
        solver = ScipyPositiveRiccatiSolver()
    else:
        raise ValueError

    X = equation.solve(solver=solver)

    assert relative_residual(A, E, B, C, R, Q, S, chol(X), trans, positive=True) < 1e-8

    for mat1, mat2 in zip(mat_old, mat_new, strict=True):
        assert type(mat1) is type(mat2)
        assert np.all(mat1 == mat2)


@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('with_Q', [False, True])
@pytest.mark.parametrize('with_S', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize(('n', 'backend'), chain(product(n_list_small, ricc_lr_backend_list_small),
                                                product(n_list_big, ricc_lr_backend_list_big)))
def test_ricc_lr(n, m, p, with_E, with_R, with_Q, with_S, trans, backend, rng):
    skip_if_missing_solver(backend)

    mat_old = []
    mat_new = []
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        mat_old.append(E.copy())
        mat_new.append(E)
    mat_old.append(A.copy())
    mat_new.append(A)
    B = rng.standard_normal((n, m))
    mat_old.append(B.copy())
    mat_new.append(B)
    C = rng.standard_normal((p, n))
    mat_old.append(C.copy())
    mat_new.append(C)
    D = rng.standard_normal((p, m))
    R0 = rng.standard_normal((m, m))
    Q0 = rng.standard_normal((p, p))
    R = D.T.dot(D) + R0.dot(R0.T) if with_R else None
    Q = D.dot(D.T) + Q0.dot(Q0.T) if with_Q else None
    if not trans:
        S = 1e-1 * D @ B.T if with_S else None
    else:
        S = 1e-1 * C.T @ D if with_S else None
    if with_R:
        mat_old.append(R.copy())
        mat_new.append(R)
    if with_Q:
        mat_old.append(Q.copy())
        mat_new.append(Q)
    if with_S:
        mat_old.append(S.copy())
        mat_new.append(S)

    equation = RiccatiEquation.from_matrices(A, E, B, C, R, Q, S, trans=trans)


    if backend == 'radi':
        from pymor.solvers.matrix_equations.radi import RADIRiccatiSolver
        solver =  RADIRiccatiSolver()
    elif backend == 'slycot':
        from pymor.bindings.slycot import SlycotRiccatiSolverLR
        solver = SlycotRiccatiSolverLR()
    elif backend == 'scipy':
        from pymor.bindings.scipy import ScipyRiccatiSolverLR
        solver = ScipyRiccatiSolverLR()
    else:
        raise ValueError

    Zva = equation.solve_lr(solver=solver)

    assert len(Zva) <= n

    Z = Zva.to_numpy()
    assert relative_residual(A, E, B, C, R, Q, S, Z, trans) < 1e-8

    for mat1, mat2 in zip(mat_old, mat_new, strict=True):
        assert type(mat1) is type(mat2)
        if sps.issparse(mat1):
            mat1 = mat1.toarray()
            mat2 = mat2.toarray()
        assert np.all(mat1 == mat2)


@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('with_Q', [False, True])
@pytest.mark.parametrize('with_S', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize(('n', 'backend'), chain(product(n_list_small, ricc_lr_backend_list_small),
                                                product(n_list_big, ricc_lr_backend_list_big)))
def test_pos_ricc_lr(n, m, p, with_E, with_R, with_Q, with_S, trans, backend, rng):
    skip_if_missing_solver(backend)

    mat_old = []
    mat_new = []
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        mat_old.append(E.copy())
        mat_new.append(E)
    mat_old.append(A.copy())
    mat_new.append(A)
    B = rng.standard_normal((n, m))
    mat_old.append(B.copy())
    mat_new.append(B)
    C = rng.standard_normal((p, n))
    mat_old.append(C.copy())
    mat_new.append(C)
    D = rng.standard_normal((p, m))
    if not trans:
        Q0 = rng.standard_normal((p, p))
        Q = D.dot(D.T) + 10 * Q0.dot(Q0.T) if with_Q else None
        S = rng.standard_normal((p, n)) if with_S else None
        R0 = rng.standard_normal((m, m))
        R = D.T.dot(D) + R0.dot(R0.T) if with_R else None
    else:
        R0 = rng.standard_normal((m, m))
        R = D.T.dot(D) + 10 * R0.dot(R0.T) if with_R else None
        S = rng.standard_normal((n, m)) if with_S else None
        Q0 = rng.standard_normal((p, p))
        Q = D.dot(D.T) + Q0.dot(Q0.T) if with_Q else None
    if with_R:
        mat_old.append(R.copy())
        mat_new.append(R)
    if with_Q:
        mat_old.append(Q.copy())
        mat_new.append(Q)
    if with_S:
        mat_old.append(S.copy())
        mat_new.append(S)

    equation = PositiveRiccatiEquation.from_matrices(A, E, B, C, R, Q, S, trans=trans)

    if backend == 'radi':
        from pymor.solvers.matrix_equations.radi import RADIPositiveRiccatiSolver
        solver = RADIPositiveRiccatiSolver()
    elif backend == 'slycot':
        from pymor.bindings.slycot import SlycotPositiveRiccatiSolverLR
        solver = SlycotPositiveRiccatiSolverLR()
    elif backend == 'scipy':
        from pymor.bindings.scipy import ScipyPositiveRiccatiSolverLR
        solver = ScipyPositiveRiccatiSolverLR()
    else:
        raise ValueError

    Zva = equation.solve_lr(solver=solver)

    assert len(Zva) <= n

    Z = Zva.to_numpy()
    assert relative_residual(A, E, B, C, R, Q, S, Z, trans, positive=True) < 1e-8

    for mat1, mat2 in zip(mat_old, mat_new, strict=True):
        assert type(mat1) is type(mat2)
        if sps.issparse(mat1):
            mat1 = mat1.toarray()
            mat2 = mat2.toarray()
        assert np.all(mat1 == mat2)

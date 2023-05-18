# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
import tempfile

import numpy as np
import pytest
import scipy.io as spio
import scipy.sparse as sps

from pymor.models.iosys import LTIModel, PHLTIModel, SecondOrderModel

pytestmark = pytest.mark.builtin


def _build_matrices_lti(with_D, with_E):
    A = sps.csc_matrix([[1, 2], [3, 4]])
    B = np.array([[1], [2]])
    C = np.array([[1, 2]])
    D = np.array([[1]]) if with_D else None
    E = np.array([[5, 6], [7, 8]]) if with_E else None
    return A, B, C, D, E


def _test_matrices_lti(A, B, C, D, E,
                       A2, B2, C2, D2, E2,
                       with_D, with_E):
    assert np.allclose(A.toarray(), A2.toarray())
    assert np.allclose(B, B2)
    assert np.allclose(C, C2)
    if with_D:
        assert np.allclose(D, D2)
    else:
        assert D2 is None
    if with_E:
        assert np.allclose(E, E2)
    else:
        assert E2 is None


def _build_matrices_phlti(with_P, with_S, with_N, with_E, with_Q):
    J = sps.csc_matrix([[0, -1], [1, 0]])
    R = sps.csc_matrix([[1, 0], [0, 1]])
    G = np.array([[1], [0]])
    P = np.array([[2], [0]]) if with_P else None
    S = np.array([[1]]) if with_S else None
    N = np.array([[0]]) if with_N else None
    E = np.array([[1, 0], [0, 2]]) if with_E else None
    Q = np.array([[2, 0], [0, 1]]) if with_Q else None
    return J, R, G, P, S, N, E, Q


def _test_matrices_phlti(J, R, G, P, S, N, E, Q,
                         J2, R2, G2, P2, S2, N2, E2, Q2,
                         with_P, with_S, with_N, with_E, with_Q):
    assert np.allclose(J.toarray(), J2.toarray())
    assert np.allclose(R.toarray(), R2.toarray())
    assert np.allclose(G, G2)

    if with_P:
        assert np.allclose(P, P2)
    else:
        assert P2 is None
    if with_S:
        assert np.allclose(S, S2)
    else:
        assert S2 is None
    if with_N:
        assert np.allclose(N, N2)
    else:
        assert N2 is None
    if with_E:
        assert np.allclose(E, E2)
    else:
        assert E2 is None
    if with_Q:
        assert np.allclose(Q, Q2)
    else:
        assert Q2 is None


def _build_matrices_so(with_Cv, with_D):
    M = sps.csc_matrix([[1, 2], [3, 4]])
    E = np.array([[5, 6], [7, 8]])
    K = np.array([[9, 10], [11, 12]])
    B = np.array([[1], [2]])
    Cp = np.array([[1, 2]])
    Cv = np.array([[3, 4]]) if with_Cv else None
    D = np.array([[1]]) if with_D else None
    return M, E, K, B, Cp, Cv, D


def _test_matrices_so(M, E, K, B, Cp, Cv, D,
                      M2, E2, K2, B2, Cp2, Cv2, D2,
                      with_Cv, with_D):
    assert np.allclose(M.toarray(), M2.toarray())
    assert np.allclose(E, E2)
    assert np.allclose(K, K2)
    assert np.allclose(B, B2)
    assert np.allclose(Cp, Cp2)
    if with_Cv:
        assert np.allclose(Cv, Cv2)
    else:
        assert Cv2 is None
    if with_D:
        assert np.allclose(D, D2)
    else:
        assert D2 is None


@pytest.mark.parametrize('with_D', [False, True])
@pytest.mark.parametrize('with_E', [False, True])
def test_matrices_lti(with_D, with_E):
    matrices = _build_matrices_lti(with_D, with_E)

    lti = LTIModel.from_matrices(*matrices)
    matrices2 = lti.to_matrices()

    _test_matrices_lti(*matrices, *matrices2, with_D, with_E)


@pytest.mark.parametrize('with_D', [False, True])
@pytest.mark.parametrize('with_E', [False, True])
def test_files_lti(with_D, with_E):
    matrices = _build_matrices_lti(with_D, with_E)

    lti = LTIModel.from_matrices(*matrices)
    with tempfile.TemporaryDirectory() as tmpdirname:
        files = (
            os.path.join(tmpdirname, 'A.mtx'),
            os.path.join(tmpdirname, 'B.mtx'),
            os.path.join(tmpdirname, 'C.mtx'),
            os.path.join(tmpdirname, 'D.mtx') if with_D else None,
            os.path.join(tmpdirname, 'E.mtx') if with_E else None,
        )
        lti.to_files(*files)
        lti2 = LTIModel.from_files(*files)
    matrices2 = lti2.to_matrices()

    _test_matrices_lti(*matrices, *matrices2, with_D, with_E)


@pytest.mark.parametrize('with_D', [False, True])
@pytest.mark.parametrize('with_E', [False, True])
def test_mat_file_lti(with_D, with_E):
    matrices = _build_matrices_lti(with_D, with_E)
    assert all(np.issubdtype(mat.dtype, np.integer) for mat in matrices if mat is not None)

    lti = LTIModel.from_matrices(*matrices)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, 'lti')
        lti.to_mat_file(file_name)
        lti2 = LTIModel.from_mat_file(file_name)
    matrices2 = lti2.to_matrices()

    _test_matrices_lti(*matrices, *matrices2, with_D, with_E)
    assert all(np.issubdtype(mat.dtype, np.floating) for mat in matrices2 if mat is not None)


def test_mat_file_lti_C():
    A, B, _, _, _ = _build_matrices_lti(False, False)

    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, 'lti')
        spio.savemat(file_name, {'A': A, 'B': B})
        lti2 = LTIModel.from_mat_file(file_name)
    matrices2 = lti2.to_matrices()

    _test_matrices_lti(A, B, B.T, None, None, *matrices2, False, False)


@pytest.mark.parametrize('with_D', [False, True])
@pytest.mark.parametrize('with_E', [False, True])
def test_abcde_files(with_D, with_E):
    matrices = _build_matrices_lti(with_D, with_E)

    lti = LTIModel.from_matrices(*matrices)
    with tempfile.TemporaryDirectory() as tmpdirname:
        files_basename = os.path.join(tmpdirname, 'lti')
        lti.to_abcde_files(files_basename)
        lti2 = LTIModel.from_abcde_files(files_basename)
    matrices2 = lti2.to_matrices()

    _test_matrices_lti(*matrices, *matrices2, with_D, with_E)


@pytest.mark.parametrize('with_P', [False, True])
@pytest.mark.parametrize('with_S', [False, True])
@pytest.mark.parametrize('with_N', [False, True])
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_Q', [False, True])
def test_matrices_phlti(with_P, with_S, with_N, with_E, with_Q):
    matrices = _build_matrices_phlti(with_P, with_S, with_N, with_E, with_Q)

    phlti = PHLTIModel.from_matrices(*matrices)
    matrices2 = phlti.to_matrices()

    _test_matrices_phlti(*matrices, *matrices2, with_P, with_S, with_N, with_E, with_Q)


@pytest.mark.parametrize('with_Cv', [False, True])
@pytest.mark.parametrize('with_D', [False, True])
def test_matrices_so(with_Cv, with_D):
    matrices = _build_matrices_so(with_Cv, with_D)

    som = SecondOrderModel.from_matrices(*matrices)
    matrices2 = som.to_matrices()

    _test_matrices_so(*matrices, *matrices2, with_Cv, with_D)


@pytest.mark.parametrize('with_Cv', [False, True])
@pytest.mark.parametrize('with_D', [False, True])
def test_files_so(with_Cv, with_D):
    matrices = _build_matrices_so(with_Cv, with_D)

    som = SecondOrderModel.from_matrices(*matrices)
    with tempfile.TemporaryDirectory() as tmpdirname:
        files = (
            os.path.join(tmpdirname, 'M.mtx'),
            os.path.join(tmpdirname, 'E.mtx'),
            os.path.join(tmpdirname, 'K.mtx'),
            os.path.join(tmpdirname, 'B.mtx'),
            os.path.join(tmpdirname, 'Cp.mtx'),
            os.path.join(tmpdirname, 'Cv.mtx') if with_Cv else None,
            os.path.join(tmpdirname, 'D.mtx') if with_D else None,
        )
        som.to_files(*files)
        som2 = SecondOrderModel.from_files(*files)
    matrices2 = som2.to_matrices()

    _test_matrices_so(*matrices, *matrices2, with_Cv, with_D)

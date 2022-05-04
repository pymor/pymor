
import numpy as np
import pytest
from pymor.algorithms.symplectic import (psd_complex_svd, psd_cotengent_lift,
                                         psd_svd_like_decomp,
                                         symplectic_gram_schmidt)
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

METHODS_DICT = {
    'psd_cotengent_lift': psd_cotengent_lift,
    'psd_complex_svd': psd_complex_svd,
    'psd_svd_like_decomp': psd_svd_like_decomp,
}
KEYS_ORTHOSYMPL_METHOD = ['psd_cotengent_lift', 'psd_complex_svd']


@pytest.mark.parametrize('key_method', METHODS_DICT.keys())
def test_symplecticity(key_method):
    """Check symplecticity of symplectic basis generation methods."""
    half_dim = 1000
    half_space = NumpyVectorSpace(half_dim)
    phase_space = BlockVectorSpace([half_space] * 2)
    n_data = 100
    U = phase_space.random(n_data, seed=42)
    modes = 10
    basis = METHODS_DICT[key_method](U, modes)
    tis_basis = basis.transposed_symplectic_inverse()
    assert np.allclose(tis_basis.to_array().inner(basis.to_array()), np.eye(modes))


@pytest.mark.parametrize('key_orthosympl_method', KEYS_ORTHOSYMPL_METHOD)
def test_orthonormality(key_orthosympl_method):
    """Check orthonormality for orthosymplectic basis generation methods."""
    half_dim = 1000
    half_space = NumpyVectorSpace(half_dim)
    phase_space = BlockVectorSpace([half_space] * 2)
    n_data = 100
    U = phase_space.random(n_data, seed=42)
    modes = 10
    basis = METHODS_DICT[key_orthosympl_method](U, modes).to_array()
    assert np.allclose(basis.inner(basis), np.eye(modes))


@pytest.mark.parametrize('test_orthonormality', [False, True])
@pytest.mark.parametrize('reiterate', [False, True])
def test_symplectic_gram_schmidt(test_orthonormality, reiterate):
    """Check symplecticity and orthonormality for symplectic_gram_schmidt."""
    half_dim = 1000
    half_space = NumpyVectorSpace(half_dim)
    phase_space = BlockVectorSpace([half_space] * 2)
    J = CanonicalSymplecticFormOperator(phase_space)
    half_red_dim = 10

    E = phase_space.random(half_red_dim, seed=42)
    if test_orthonormality:
        # special choice, such that result is orthosymplectic
        F = J.apply(E)
    else:
        # less structure in snapshots, no orthogonality
        F = phase_space.random(half_red_dim, seed=43)

    S, Lambda = symplectic_gram_schmidt(E, F, return_Lambda=True, reiterate=reiterate)
    tsi_S = S.transposed_symplectic_inverse()
    arr_S = S.to_array()
    arr_tsi_S = tsi_S.to_array()
    # symplecticity
    assert np.allclose(arr_tsi_S.inner(arr_S), np.eye(len(arr_S)))
    if test_orthonormality:
        # orthogonality (due to special choice of E and F in symplectic_gram_schmidt)
        assert np.allclose(arr_S.inner(arr_S), np.eye(len(arr_S)))
    # check tsi_S.T * M = Lambda
    M = E.copy()
    M.append(F)
    assert np.allclose(arr_tsi_S.inner(M), Lambda)
    # check M = S * Lambda
    assert np.allclose(arr_S.lincomb(Lambda.T).to_numpy(), M.to_numpy())
    # upper triangular of permuted matrx
    n = Lambda.shape[0] // 2
    perm = np.vstack([np.arange(0, n), np.arange(n, 2*n)]).T.reshape(-1)
    assert len(set(perm)) == len(perm) == 2*n
    assert np.allclose(np.tril(Lambda[perm, :][:, perm], k=-1), np.zeros_like(Lambda))

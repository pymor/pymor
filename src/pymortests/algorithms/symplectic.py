
import numpy as np
from pymor.algorithms.symplectic import (psd_complex_svd, psd_cotengent_lift,
                                         psd_svd_like_decomp,
                                         symplectic_gram_schmidt)
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.block import BlockVectorSpace


def test_symplecticity():
    """Check symplecticity of symplectic basis generation methods."""
    basis_gen_methods = [psd_cotengent_lift, psd_complex_svd, psd_svd_like_decomp]

    half_dim = 1000
    half_space = NumpyVectorSpace(half_dim)
    phase_space = BlockVectorSpace([half_space] * 2)
    n_data = 100
    U = phase_space.random(n_data)
    modes = 10
    for method in basis_gen_methods:
        basis = method(U, modes)
        tis_basis = basis.transposed_symplectic_inverse()
        assert np.allclose(tis_basis.to_array().inner(basis.to_array()), np.eye(modes))


def test_orthonormality():
    """Check orthonormality for orthosymplectic basis generation methods."""
    basis_gen_methods = [psd_cotengent_lift, psd_complex_svd]

    half_dim = 1000
    half_space = NumpyVectorSpace(half_dim)
    phase_space = BlockVectorSpace([half_space] * 2)
    n_data = 100
    U = phase_space.random(n_data)
    modes = 10
    for method in basis_gen_methods:
        basis = method(U, modes).to_array()
        assert np.allclose(basis.inner(basis), np.eye(modes))


def test_symplectic_gram_schmidt():
    """Check symplecticity and orthonormality for symplectic_gram_schmidt."""
    half_dim = 1000
    half_space = NumpyVectorSpace(half_dim)
    phase_space = BlockVectorSpace([half_space] * 2)
    J = CanonicalSymplecticFormOperator(phase_space)
    half_red_dim = 10

    def run_test(E, F, test_orthonormality):
        for reiterate in (False, True):  # without and with reorthogonalization
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

    # special choice, such that result is orthosymplectic
    E = phase_space.random(half_red_dim)
    run_test(E, J.apply(E), test_orthonormality=True)
    # less structure in snapshots, no orthogonality
    F = phase_space.random(half_red_dim)
    run_test(E, F, test_orthonormality=False)

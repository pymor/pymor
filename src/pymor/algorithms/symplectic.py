# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.linalg import schur

from pymor.algorithms.pod import pod
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.interface import VectorArray


class SymplecticBasis(BasicObject):
    """ A canonically |SymplecticBasis| based on pairs of basis vectors (e_i, f_i).

    The basis vectors are each contained in a |VectorArray|

        E = (e_i)_{i=1}^n
        F = (f_i)_{i=1}^n

    such that

        V = [E, F].

    Parameters
    ----------
    J
        A |CanonicalSymplecticFormOperator|.
    E
        A |VectorArray| that represents the first half of basis vectors.
    F
        A |VectorArray| that represents the second half of basis vectors.
    """
    def __init__(self, J, E=None, F=None):
        if E is None:
            E = J.source.empty()
        if F is None:
            F = J.source.empty()
        assert isinstance(J, CanonicalSymplecticFormOperator)
        assert isinstance(E, VectorArray)
        assert isinstance(F, VectorArray)
        assert E.space == F.space and len(E) == len(F)
        assert J.source == J.range == E.space
        self.phase_space = J.source
        self.E = E
        self.F = F
        self.J = J

    @classmethod
    def from_array(self, U, J):
        """ Generate |SymplecticBasis| from |VectorArray|.

        Parameters
        ----------
        U
            The |VectorArray|.

        Returns
        -------
        BASIS
            The |SymplecticBasis|.
        """
        assert len(U) % 2 == 0, 'the symplectic array has to be even-dimensional'
        return SymplecticBasis(
            J,
            U[:len(U)//2],
            U[len(U)//2:],
        )

    def transposed_symplectic_inverse(self):
        """ Compute transposed symplectic inverse J_{2N}.T * V * J_{2n}.

        Returns
        -------
        TSI_BASIS
            The transposed symplectic inverse as |SymplecticBasis|.
        """
        E = self.J.apply_adjoint(self.F*(-1))
        F = self.J.apply_adjoint(self.E)
        return SymplecticBasis(self.J, E, F)

    def to_array(self):
        """ Convert to |VectorArray|.

        Returns
        -------
        BASIS
            The |SymplecticBasis| as |VectorArray|.
        """
        U = self.E.copy()
        U.append(self.F)
        return U

    def __len__(self):
        assert len(self.E) == len(self.F)
        return 2*len(self.E)

    def append(self, other, remove_from_other=False):
        """ Append another |SymplecticBasis|.

        other
            The |SymplecticBasis| to append.
        remove_from_other
            Flag, wether to remove vectors from other.
        """
        assert isinstance(other, SymplecticBasis)
        assert other.phase_space == self.phase_space
        self.E.append(other.E, remove_from_other)
        self.F.append(other.F, remove_from_other)

    def check_symplecticity(self, offset=0, check_tol=1e-3):
        """ Check symplecticity w.r.t. a given transposed symplectic inverse.

        Parameters
        ----------
        offset
            Used in iterative methods. Needs to be even.
        check_tol
            tolerance for which method returns true.
        """
        assert offset % 2 == 0

        h_off = offset//2
        h_len = len(self)//2
        idx = np.arange(h_off, h_len)

        tsi_self = self.transposed_symplectic_inverse()

        error_matrix = tsi_self[idx].to_array().inner(self.to_array())
        error_matrix[:, np.hstack([idx, h_len+idx])] -= np.eye(len(self) - offset)
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError(f"result not symplectic (max err={err})")

    def __getitem__(self, ind):
        assert self.E.check_ind(ind)
        return type(self)(self.J, self.E[ind], self.F[ind])

    def lincomb(self, coefficients):
        assert isinstance(coefficients, np.ndarray)
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, ...]
        assert len(coefficients.shape) == 2 and coefficients.shape[1] == len(self)
        result = self.E.lincomb(coefficients[:, :len(self.E)])
        result += self.F.lincomb(coefficients[:, len(self.E):])
        return result

    def extend(self, U, method='svd_like', modes=2, product=None):
        """ Extend the |SymplecticBasis| with vectors from a |VectorArray|.

        Parameters
        ----------
        U
            The vectors used for the extension as |VectorArray|.
        method
            The method used for extension. Available options are
                ('svd_like', 'complex_svd', 'symplectic_gram_schmidt').
        modes
            Number of modes to extract from U. Has to be even.
        product
            A product to use for the projection error. Default is None.
        """
        from pymor.algorithms.symplectic import (psd_complex_svd,
                                                 psd_svd_like_decomp,
                                                 symplectic_gram_schmidt)
        assert modes % 2 == 0, 'number of modes has to be even'
        assert method in ('svd_like', 'complex_svd', 'symplectic_gram_schmidt')

        U_proj_err = U - self.lincomb(U.inner(self.transposed_symplectic_inverse().to_array()))
        proj_error = U_proj_err.norm(product=product)

        if method in ('svd_like', 'complex_svd'):

            if method == 'svd_like':
                new_basis = psd_svd_like_decomp(U_proj_err, self.J, modes)
            elif method == 'complex_svd':
                new_basis = psd_complex_svd(U_proj_err, self.J, modes)

            self.append(new_basis)

        elif method == 'symplectic_gram_schmidt':
            basis_length = len(self)
            # find max error
            idx = proj_error.argsort()[-modes//2:][::-1]
            new_basis = U[idx].copy()
            new_basis.scal(1/new_basis.norm())
            new_basis.append(self.J.apply_adjoint(new_basis))
            self.append(SymplecticBasis.from_array(new_basis, self.J))
            symplectic_gram_schmidt(self.E, self.F, self.J, offset=basis_length, copy=False)


def psd_svd_like_decomp(U, J, modes, balance=True):
    """ Generates a |SymplecticBasis| with the PSD SVD-like decompostion.

    Parameters
    ----------
    U
        The |VectorArray| for which the PSD SVD-like decompostion is to be computed.
    J
        A |CanonicalSymplecticFormOperator| operating on the same |VectorSpace| as U.
    modes
        Number of modes (needs to be even).
    balance
        A flag, wether to balance the norms of pairs of basis vectors.

    Returns
    -------
    BASIS
        The |SymplecticBasis|.
    """
    assert modes % 2 == 0
    assert U.dim % 2 == 0
    if not isinstance(J, CanonicalSymplecticFormOperator):
        raise NotImplementedError
    assert U in J.source

    symplectic_gramian = U.gramian(J)

    DJD, Q, _ = schur(symplectic_gramian, sort=lambda x: x.imag > 0)
    i_sort = range(0, modes)
    i_sort = np.hstack([i_sort[::2], i_sort[1::2]])
    Q = Q[:, i_sort]
    DJD = DJD[:, i_sort][i_sort, :]
    inv_D = 1 / np.sqrt(np.abs(np.diag(DJD[(modes//2):, :(modes//2)])))
    inv_D = np.hstack([inv_D, -inv_D*np.sign(np.diag(DJD[(modes//2):, :(modes//2)]))])
    S = U.lincomb((Q * inv_D[np.newaxis, :]).T)

    # balance norms of basis vector pairs s_i, s_{modes+1}
    # with a symplectic, orthogonal transformation
    if balance:
        a = S.norm2()
        a = a[:modes//2] - a[modes//2:]
        b = 2*S[:modes//2].pairwise_inner(S[modes//2:])
        c = np.sqrt(a**2 + b**2)
        phi = np.vstack([a+c-b, a+c+b])
        norm_phi = np.sqrt(np.sum(phi**2, axis=0))
        phi = phi / norm_phi
        balance_coeff = np.block([
            [np.diag(phi[0, :]), -np.diag(phi[1, :])],
            [np.diag(phi[1, :]), np.diag(phi[0, :])]
        ])
        S = S.lincomb(balance_coeff.T)

    return SymplecticBasis.from_array(S, J)


def psd_cotengent_lift(U, J, modes):
    """ Generates a |SymplecticBasis| with the PSD cotangent lift.

    Parameters
    ----------
    U
        The |VectorArray| for which the PSD SVD-like decompostion is to be computed.
    J
        A |CanonicalSymplecticFormOperator| operating on the same |VectorSpace| as U.
    modes
        Number of modes (needs to be even).

    Returns
    -------
    BASIS
        The |SymplecticBasis|.
    """
    assert isinstance(J, CanonicalSymplecticFormOperator)
    assert isinstance(U.space, BlockVectorSpace) and len(U.space.subspaces) == 2 and \
        U.space.subspaces[0] == U.space.subspaces[1]
    assert modes % 2 == 0
    assert U in J.source

    X = U.block(0).copy()
    X.append(U.block(1).copy())
    V, svals = pod(X, modes=modes // 2)

    return SymplecticBasis(
        J,
        J.source.make_array([V, J.source.subspaces[1].zeros(len(V))]),
        J.source.make_array([J.source.subspaces[0].zeros(len(V)), V]),
    )


def psd_complex_svd(U, J, modes):
    """ Generates a |SymplecticBasis| with the PSD complex SVD.

    Parameters
    ----------
    U
        The |VectorArray| for which the PSD SVD-like decompostion is to be computed.
    J
        A |CanonicalSymplecticFormOperator| operating on the same |VectorSpace| as U.
    modes
        Number of modes (needs to be even).

    Returns
    -------
    BASIS
        The |SymplecticBasis|.
    """
    assert isinstance(J, CanonicalSymplecticFormOperator)
    assert isinstance(U.space, BlockVectorSpace) and len(U.space.subspaces) == 2 and \
        U.space.subspaces[0] == U.space.subspaces[1]
    assert modes % 2 == 0
    assert U in J.source

    X = U.block(0) + U.block(1) * 1j

    V, _ = pod(X, modes=modes // 2)

    return SymplecticBasis(
        J,
        J.source.make_array([V.real, V.imag]),
        J.source.make_array([-V.imag, V.real]),
    )


@defaults('atol', 'rtol', 'check', 'check_tol')
def symplectic_gram_schmidt(E, F, J, return_Lambda=False, atol=1e-13, rtol=1e-13, offset=0,
                            lmax=2, check=True, check_tol=1e-3, copy=True):
    """ Symplectify a |VectorArray| using the modified symplectic Gram-Schmidt algorithm.

    Reference::

        Salam (2005), On theoretical and numerical aspects of symplectic Gram--Schmidt-like
        algorithms

    Decomposition::

        [E, F] = S * Lambda

    with S symplectic and Lambda a permuted upper-triangular matrix.

    Parameters
    ----------
    E, F
        The two |VectorArray| which are to be symplectified.
    J
        An |Operator| describing the symplectic Form.
    return_Lambda
        If `True`, the matrix `Lambda` from the decomposition is returned.
    atol
        Vectors of norm smaller than `atol` are removed from the array.
    rtol
        Relative tolerance used to detect non-symplectic subspaces
        (which are then removed from the array).
    offset
        Assume that the first `offset` pairs vectors in E and F are already symplectic and start the
        algorithm at the `offset + 1`-th vector.
    lmax
        Number of symplectification iterations.
            lmax = 1: modified symplectic Gram-Schmidt algorithm,
            lmax = 2: modified symplectic Gram-Schmidt algorithm with reorthogonalization.
    check
        If `True`, check if the resulting |VectorArray| is really symplectic.
    check_tol
        Tolerance for the check.

    Returns
    -------
    S
        The symplectified |VectorArray|.
    Lambda
        if `return_Lambda` is `True`.
    """
    assert isinstance(J, CanonicalSymplecticFormOperator), NotImplementedError
    assert E.space == F.space
    assert len(E) == len(F)
    assert E.dim % 2 == 0
    assert lmax in (1, 2)
    assert offset % 2 == 0

    logger = getLogger('pymor.algorithms.symplectic_gram_schmidt.symplectic_gram_schmidt')

    if copy:
        E = E.copy()
        F = F.copy()

    # main loop
    p = len(E)
    J2T = np.array([
        [0, -1],
        [1, 0]
    ])
    Lambda = np.zeros((2*p, 2*p))
    remove = []  # indices of to be removed vectors
    for j in range(offset//2, p):
        # first calculate symplecticity value
        initial_sympl = abs(J.apply2(E[j], F[j]))

        if initial_sympl < atol:
            logger.info(f"Removing vector pair {j} with symplecticity value {initial_sympl}")
            remove.append(j)
            continue

        # lmax = 1: MSGS
        # lmax = 2: MSGSR
        for _ in range(lmax):
            # symplectify to all vectors left
            for i in range(j):
                if i in remove:
                    continue
                P = J2T @ J.apply2(cat_arrays([E[i], F[i]]), cat_arrays([E[j], F[j]]))
                E[j].axpy(-P[0, 0], E[i])
                F[j].axpy(-P[1, 1], F[i])
                E[j].axpy(-P[1, 0], F[i])
                F[j].axpy(-P[0, 1], E[i])

                Lambda[np.ix_([i, p+i], [j, p+j])] += P

        # remove vector pair if it got a too small symplecticty value
        if abs(J.apply2(E[j], F[j])) < rtol * initial_sympl:
            logger.info(f"Removing vector pair {j} due to small symplecticty value")
            remove.append(j)
            continue

        Lambda[np.ix_([j, p+j], [j, p+j])] = esr(E[j], F[j], J)

    if remove:
        del E[remove]
        del F[remove]
        remove = np.array(remove)
        Lambda = np.delete(Lambda, p + remove, axis=0)
        Lambda = np.delete(Lambda, remove, axis=0)

    S = SymplecticBasis(J, E, F)
    if check:
        S.check_symplecticity(offset=offset, check_tol=check_tol)

    if return_Lambda:
        return S, Lambda
    else:
        return S


def esr(E, F, J):
    """ Elemenraty SR factorization. Transforms E and F such that

        [E, F] = S * diag(r11, r22)

    Coefficients are chosen such that ||E|| = ||F||. r12 is set to zero.

    Parameters
    ----------
    E
        A |VectorArray| of dim=1 from the same |VectorSpace| as F.
    F
        A |VectorArray| of dim=1 from the same |VectorSpace| as E.
    J
        A |CanonicalSymplecticFormOperator| operating on the same |VectorSpace| as E and F.

    Returns
    -------
    R
        A diagonal numpy.ndarray.
    """
    assert E in J.source
    assert F in J.source
    assert len(E) == len(F) == 1

    sympl_coeff = J.apply2(E, F).item()
    r11 = np.sqrt(E.norm().item() / F.norm().item() * abs(sympl_coeff)) * np.sign(sympl_coeff)
    E.scal(1 / r11)
    r22 = sympl_coeff / r11
    F.scal(1 / r22)

    return np.array([
        [r11, 0],
        [0, r22]
    ])

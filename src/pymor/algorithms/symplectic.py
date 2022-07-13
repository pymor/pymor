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
from pymor.vectorarrays.interface import VectorArray, VectorSpace


class SymplecticBasis(BasicObject):
    """A canonically-symplectic basis based on pairs of basis vectors (e_i, f_i).

    Is either initialzed (a) with a pair of |VectorArrays| E and F or (b) with a |VectorSpace|.
    The basis vectors are each contained in a |VectorArray|

        E = (e_i)_{i=1}^n
        F = (f_i)_{i=1}^n

    such that

        V = [E, F].

    Parameters
    ----------
    E
        A |VectorArray| that represents the first half of basis vectors. May be none if phase_space
        is specified.
    F
        A |VectorArray| that represents the second half of basis vectors. May be none if phase_space
        is specified.
    phase_space
        A |VectorSpace| that represents the phase space. May be none if E and F are specified.
    check_symplecticity
        Flag, wether to check symplecticity of E and F in the constructor (if these are not None).
        Default is True.
    """

    def __init__(self, E=None, F=None, phase_space=None, check_symplecticity=True):
        if phase_space is None:
            assert E is not None and F is not None
            phase_space = E.space
        if E is None:
            E = phase_space.empty()
        if F is None:
            F = phase_space.empty()
        assert isinstance(phase_space, VectorSpace)
        assert isinstance(E, VectorArray)
        assert isinstance(F, VectorArray)
        assert E.space == F.space == phase_space and len(E) == len(F)
        self.__auto_init(locals())

        if check_symplecticity and len(E) > 0:
            self._check_symplecticity()

    @classmethod
    def from_array(self, U, check_symplecticity=True):
        """Generate |SymplecticBasis| from |VectorArray|.

        Parameters
        ----------
        U
            The |VectorArray|.
        check_symplecticity
            Flag, wether to check symplecticity of E and F in the constructor (if these are not
            None). Default is True.

        Returns
        -------
        BASIS
            The |SymplecticBasis|.
        """
        assert len(U) % 2 == 0, 'the symplectic array has to be even-dimensional'
        return SymplecticBasis(
            U[:len(U)//2],
            U[len(U)//2:],
            check_symplecticity=check_symplecticity,
        )

    def transposed_symplectic_inverse(self):
        """Compute transposed symplectic inverse J_{2N}.T * V * J_{2n}.

        Returns
        -------
        TSI_BASIS
            The transposed symplectic inverse as |SymplecticBasis|.
        """
        J = CanonicalSymplecticFormOperator(self.phase_space)
        E = J.apply_adjoint(self.F*(-1))
        F = J.apply_adjoint(self.E)
        # check_symplecticity = False, otherwise recursion loop
        return SymplecticBasis(E, F, check_symplecticity=False)

    def to_array(self):
        """Convert to |VectorArray|.

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
        return len(self.E)

    def append(self, other, remove_from_other=False, check_symplecticity=True):
        """Append another |SymplecticBasis|.

        other
            The |SymplecticBasis| to append.
        remove_from_other
            Flag, wether to remove vectors from other.
        check_symplecticity
            Flag, wether to check symplecticity of E and F in the constructor (if these are not
            None). Default is True.
        """
        assert isinstance(other, SymplecticBasis)
        assert other.phase_space == self.phase_space
        old_len = len(self)
        self.E.append(other.E, remove_from_other)
        self.F.append(other.F, remove_from_other)

        if check_symplecticity and len(self.E) > 0:
            # skip vectors which were already in the basis before append
            self._check_symplecticity(offset=old_len)

    def _check_symplecticity(self, offset=0, check_tol=1e-3):
        """Check symplecticity of the |SymplecticBasis|.

        Parameters
        ----------
        offset
            Can be used to offset the check of symplecicity to the basis vectors with index bigger
            than the offset. This is useful in iterative methods to avoid checking multiple times.
            The offset needs to be even. The default value is 0, i.e. all basis vectors are checked
            by default.
        check_tol
            tolerance for which an error is raised.

        Raises
        ------
        AccuracyError
            Is raised when the symplecicity for some pair (e_i, f_i) exceeds check_tol.
        """
        idx = np.arange(offset, len(self))

        tsi_self = self.transposed_symplectic_inverse()

        error_matrix = tsi_self[idx].to_array().inner(self.to_array())
        error_matrix[:, np.hstack([idx, len(self)+idx])] -= np.eye((len(self) - offset)*2)
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError(f"result not symplectic (max err={err})")

    def __getitem__(self, ind):
        assert self.E.check_ind(ind)
        # check_symplecticity = False, otherwise recursion loop
        return type(self)(self.E[ind], self.F[ind], check_symplecticity=False)

    def lincomb(self, coefficients):
        assert isinstance(coefficients, np.ndarray)
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, ...]
        assert len(coefficients.shape) == 2 and coefficients.shape[1] == 2*len(self)
        result = self.E.lincomb(coefficients[:, :len(self)])
        result += self.F.lincomb(coefficients[:, len(self):])
        return result

    def extend(self, U, method='svd_like', modes=2, product=None):
        """Extend the |SymplecticBasis| with vectors from a |VectorArray|.

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
                new_basis = psd_svd_like_decomp(U_proj_err, modes)
            elif method == 'complex_svd':
                new_basis = psd_complex_svd(U_proj_err, modes)

            self.append(new_basis)

        elif method == 'symplectic_gram_schmidt':
            J = CanonicalSymplecticFormOperator(self.phase_space)
            basis_length = len(self)
            # find max error
            idx = proj_error.argsort()[-modes//2:][::-1]
            new_basis = U[idx].copy()
            new_basis.scal(1/new_basis.norm())
            new_basis.append(J.apply_adjoint(new_basis))
            self.append(
                SymplecticBasis.from_array(new_basis, check_symplecticity=False),
                check_symplecticity=False,
            )
            symplectic_gram_schmidt(self.E, self.F, offset=basis_length, copy=False)
        else:
            assert False


def psd_svd_like_decomp(U, modes, balance=True):
    """Generates a |SymplecticBasis| with the PSD SVD-like decompostion.

    This is an implementation of Algorithm 1 in :cite:`BBH19`.

    Parameters
    ----------
    U
        The |VectorArray| for which the PSD SVD-like decompostion is to be computed.
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

    J = CanonicalSymplecticFormOperator(U.space)
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

    return SymplecticBasis.from_array(S)


def psd_cotengent_lift(U, modes):
    """Generates a |SymplecticBasis| with the PSD cotangent lift.

    This is an implementation of Algorithm 1 in :cite:`PM16`.

    Parameters
    ----------
    U
        The |VectorArray| for which the PSD SVD-like decompostion is to be computed.
    modes
        Number of modes (needs to be even).

    Returns
    -------
    BASIS
        The |SymplecticBasis|.
    """
    assert isinstance(U.space, BlockVectorSpace) and len(U.space.subspaces) == 2 and \
        U.space.subspaces[0] == U.space.subspaces[1]
    assert modes % 2 == 0

    X = U.blocks[0].copy()
    X.append(U.blocks[1].copy())
    V, svals = pod(X, modes=modes // 2)

    return SymplecticBasis(
        U.space.make_array([V, V.space.zeros(len(V))]),
        U.space.make_array([V.space.zeros(len(V)), V]),
    )


def psd_complex_svd(U, modes):
    """Generates a |SymplecticBasis| with the PSD complex SVD.

    This is an implementation of Algorithm 2 in :cite:`PM16`.

    Parameters
    ----------
    U
        The |VectorArray| for which the PSD SVD-like decompostion is to be computed.
    modes
        Number of modes (needs to be even).

    Returns
    -------
    BASIS
        The |SymplecticBasis|.
    """
    assert isinstance(U.space, BlockVectorSpace) and len(U.space.subspaces) == 2 and \
        U.space.subspaces[0] == U.space.subspaces[1]
    assert modes % 2 == 0

    X = U.blocks[0] + U.blocks[1] * 1j

    V, _ = pod(X, modes=modes // 2)

    return SymplecticBasis(
        U.space.make_array([V.real, V.imag]),
        U.space.make_array([-V.imag, V.real]),
    )


@defaults('atol', 'rtol', 'reiterate', 'reiteration_threshold', 'check', 'check_tol')
def symplectic_gram_schmidt(E, F, return_Lambda=False, atol=1e-13, rtol=1e-13, offset=0,
                            reiterate=True, reiteration_threshold=9e-1, check=True, check_tol=1e-3,
                            copy=True):
    """Symplectify a |VectorArray| using the modified symplectic Gram-Schmidt algorithm.

    This is an implementation of Algorithm 3.2. in :cite:`S11` with a modified criterion for
    reiteration.

    Decomposition::

        [E, F] = S * Lambda

    with S symplectic and Lambda a permuted upper-triangular matrix.

    Parameters
    ----------
    E, F
        The two |VectorArrays| which are to be symplectified.
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
    reiterate
        If `True`, symplectify again if the symplectic product of the symplectified vectors
        is much smaller than the symplectic product of the original vector.
    reiteration_threshold
        If `reiterate` is `True`, "re-orthonormalize" if the ratio between the symplectic
        products of the symplectified vectors and the original vectors is smaller than this value.
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
    assert E.space == F.space
    assert len(E) == len(F)
    assert E.dim % 2 == 0

    logger = getLogger('pymor.algorithms.symplectic_gram_schmidt.symplectic_gram_schmidt')

    if copy:
        E = E.copy()
        F = F.copy()

    J = CanonicalSymplecticFormOperator(E.space)

    # main loop
    p = len(E)
    J2T = np.array([
        [0, -1],
        [1, 0]
    ])
    Lambda = np.zeros((2*p, 2*p))
    remove = []  # indices of to be removed vectors
    for j in range(offset, p):
        # first calculate symplecticity value
        initial_sympl = abs(J.apply2(E[j], F[j]))

        if initial_sympl < atol:
            logger.info(f"Removing vector pair {j} with symplecticity value {initial_sympl}")
            remove.append(j)
            continue

        sympl = initial_sympl
        while True:
            # symplectify to all vectors left
            for i in range(j):
                if i in remove:
                    continue
                P = J2T @ np.block([
                    [J.apply2(E[i], E[j]), J.apply2(E[i], F[j])],
                    [J.apply2(F[i], E[j]), J.apply2(F[i], F[j])],
                ])
                E[j].axpy(-P[0, 0], E[i])
                F[j].axpy(-P[1, 1], F[i])
                E[j].axpy(-P[1, 0], F[i])
                F[j].axpy(-P[0, 1], E[i])

                Lambda[np.ix_([i, p+i], [j, p+j])] += P

            # calculate new symplectic product
            old_sympl, sympl = sympl, abs(J.apply2(E[j], F[j]))

            # remove vector pair if it got a too small symplecticty value
            if sympl < rtol * initial_sympl:
                logger.info(f"Removing vector pair {j} due to small symplecticty value")
                remove.append(j)
                break

            # check if reorthogonalization should be done
            if reiterate and sympl < reiteration_threshold * old_sympl:
                logger.info(f"Symplectifying vector pair {j} again")
            else:
                Lambda[np.ix_([j, p+j], [j, p+j])] = esr(E[j], F[j], J)
                break

    if remove:
        del E[remove]
        del F[remove]
        remove = np.array(remove)
        Lambda = np.delete(Lambda, p + remove, axis=0)
        Lambda = np.delete(Lambda, remove, axis=0)

    S = SymplecticBasis(E, F)
    if check:
        S._check_symplecticity(offset=offset, check_tol=check_tol)

    if return_Lambda:
        return S, Lambda
    else:
        return S


def esr(E, F, J=None):
    """Elemenraty SR factorization. Transforms E and F such that

        [E, F] = S * diag(r11, r22)

    Coefficients are chosen such that ||E|| = ||F||. r12 is set to zero.

    Parameters
    ----------
    E
        A |VectorArray| of dim=1 from the same |VectorSpace| as F.
    F
        A |VectorArray| of dim=1 from the same |VectorSpace| as E.
    J
        A |CanonicalSymplecticFormOperator| operating on the same |VectorSpace| as E and F. Default
        is CanonicalSymplecticFormOperator(E.space).

    Returns
    -------
    R
        A diagonal numpy.ndarray.
    """
    if J is None:
        J = CanonicalSymplecticFormOperator(E.space)
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

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger


@defaults('atol', 'rtol', 'reiterate', 'reiteration_threshold', 'check', 'check_tol')
def gram_schmidt(A, product=None, return_R=False, atol=1e-13, rtol=1e-13, offset=0,
                 reiterate=True, reiteration_threshold=9e-1, check=True, check_tol=1e-3,
                 copy=True):
    """Orthonormalize a |VectorArray| using the modified Gram-Schmidt algorithm.

    Parameters
    ----------
    A
        The |VectorArray| which is to be orthonormalized.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    return_R
        If `True`, the R matrix from QR decomposition is returned.
    atol
        Vectors of norm smaller than `atol` are removed from the array.
    rtol
        Relative tolerance used to detect linear dependent vectors
        (which are then removed from the array).
    offset
        Assume that the first `offset` vectors are already orthonormal and start the
        algorithm at the `offset + 1`-th vector.
    reiterate
        If `True`, orthonormalize again if the norm of the orthogonalized vector is
        much smaller than the norm of the original vector.
    reiteration_threshold
        If `reiterate` is `True`, re-orthonormalize if the ratio between the norms of
        the orthogonalized vector and the original vector is smaller than this value.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.

    Returns
    -------
    Q
        The orthonormalized |VectorArray|.
    R
        The upper-triangular/trapezoidal matrix (if `return_R` is `True`).
    """
    logger = getLogger('pymor.algorithms.gram_schmidt.gram_schmidt')

    if copy:
        A = A.copy()

    # main loop
    R = np.eye(len(A))
    remove = []  # indices of to be removed vectors
    for i in range(offset, len(A)):
        # first calculate norm
        initial_norm = A[i].norm(product)[0]

        if initial_norm <= atol:
            logger.info(f'Removing vector {i} of norm {initial_norm}')
            remove.append(i)
            continue

        if i == 0:
            A[0].scal(1 / initial_norm)
            R[i, i] = initial_norm
        else:
            norm = initial_norm
            # If reiterate is True, reiterate as long as the norm of the vector changes
            # strongly during orthogonalization (due to Andreas Buhr).
            while True:
                # orthogonalize to all vectors left
                for j in range(i):
                    if j in remove:
                        continue
                    p = A[j].pairwise_inner(A[i], product)[0]
                    A[i].axpy(-p, A[j])
                    common_dtype = np.promote_types(R.dtype, type(p))
                    R = R.astype(common_dtype, copy=False)
                    R[j, i] += p

                # calculate new norm
                old_norm, norm = norm, A[i].norm(product)[0]

                # remove vector if it got too small
                if norm <= rtol * initial_norm:
                    logger.info(f'Removing linearly dependent vector {i}')
                    remove.append(i)
                    break

                # check if reorthogonalization should be done
                if reiterate and norm < reiteration_threshold * old_norm:
                    logger.info(f'Orthonormalizing vector {i} again')
                else:
                    A[i].scal(1 / norm)
                    R[i, i] = norm
                    break

    if remove:
        del A[remove]
        R = np.delete(R, remove, axis=0)

    if check:
        error_matrix = A[offset:len(A)].inner(A, product)
        error_matrix[:len(A) - offset, offset:len(A)] -= np.eye(len(A) - offset)
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError(f'result not orthogonal (max err={err})')

    if return_R:
        return A, R
    else:
        return A


@defaults('atol', 'rtol', 'check', 'check_tol')
def cgs_iro_ls(A, product=None, return_R=False, atol=1E-13, rtol=1e-13, offset=0,
               check=True, check_tol=1e-3, copy=True):
    """Orthonormalize a |VectorArray| using the CGS IRO LS algortihm.

    This method computes a QR decomposition of a |VectorArray| via the classical Gram-Schmidt
    algorithm with normalization lag and re-orthogonalization lag according to :cite:`SLAYT21`.
    Derived from the matlab repository :cite:`LCO24` to include tolerance checks,
    offsets and non-euclidean inner products.

    Parameters
    ----------
    A
        The |VectorArray| which is to be orthonormalized.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    return_R
        If `True`, the R matrix from QR decomposition is returned.
    atol
        Vectors of norm smaller than `atol` are removed from the array.
    rtol
        Relative tolerance used to detect linear dependent vectors
        (which are then removed from the array).
    offset
        Assume that the first `offset` vectors are already orthonormal and start the
        algorithm at the `offset + 1`-th vector.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.

    Returns
    -------
    Q
        The orthonormalized |VectorArray|.
    R
        The upper-triangular/trapezoidal matrix (if `return_R` is `True`).
    """
    logger = getLogger('pymor.algorithms.gram_schmidt.cgs_iro_ls')

    n = len(A)
    assert 0 <= atol
    assert 0 <= offset <= n
    if copy:
        A = A.copy()

    atol_p2 = atol ** 2

    R = np.eye(n)
    remove = []
    r0 = r1 = None

    if n == 0 or offset == n:
        return (A, R) if return_R else A

    # last two orthonormal vectors have to be orthogonalized again
    # to achieve an orthonormal basis
    offset = max(offset-2, 0)

    for k in range(offset, n-1):
        q = A[k]
        u = A[k+1]

        r0 = q.inner(q, product)[0][0]
        initial_norm = np.sqrt(r0)
        if r0 <= atol_p2:
            logger.info(f'Removing vector {k} of norm {r0}')
            remove.append(k)
            continue
        r1 = q.inner(u, product)[0][0]

        if k > offset:
            y = np.reshape(A[:k].inner(q, product), [k])
            z = np.reshape(A[:k].inner(u, product), [k])
            r0 -= y.dot(y)
            r1 -= y.dot(z)
            R[:k, k] += y

        if r0 <= rtol * initial_norm:
            logger.info(f'Removing linearly dependent vector {k}')
            remove.append(k)
            continue

        r0 = np.sqrt(r0)

        common_dtype = np.promote_types(R.dtype, type(r0))
        R = R.astype(common_dtype, copy=False)

        R[k, k] = r0
        R[k, k+1] = r1 / r0

        if k > offset:
            R[:k, k+1] = z
            q.axpy(-1.0, A[:k].lincomb(y))

        q.scal(1 / r0)
        u.axpy(-1.0, A[:k+1].lincomb(R[:k+1, k+1]))

    # orth last vector outside loop
    q = A[n-1]
    r0 = q.inner(q, product)[0][0]
    initial_norm = np.sqrt(r0)
    if r0 <= atol_p2 or r0 <= rtol * initial_norm:
        logger.info(f'Removing vector {n-1} of norm {r0}')
        remove.append(n-1)
    else:
        y = np.reshape(A[:n-1].inner(q, product), [n-1]) if n>1 else np.zeros([0])

        common_dtype = np.promote_types(R.dtype, type(r0))
        R = R.astype(common_dtype, copy=False)

        R[:n-1, n-1] += y

        r0 -= y.dot(y)
        if r0 <= rtol * initial_norm:
            logger.info(f'Removing linearly dependent vector {k}')
            remove.append(n-1)
        else:
            r0 = np.sqrt(r0)
            R[n-1, n-1] = r0

            q.axpy(-1.0, A[:n-1].lincomb(y))
            q.scal(1 / r0)

    if remove:
        del A[remove]
        R = np.delete(R, remove, axis=0)

    if check:
        error_matrix = A[offset:len(A)].inner(A, product)
        error_matrix[:len(A) - offset, offset:len(A)] -= np.eye(len(A) - offset)
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError(f'Result not orthogonal (max err={err})')

    return (A, R) if return_R else A


def gram_schmidt_biorth(V, W, product=None,
                        reiterate=True, reiteration_threshold=1e-1, check=True, check_tol=1e-3,
                        copy=True):
    """Biorthonormalize a pair of |VectorArrays| using the biorthonormal Gram-Schmidt process.

    See Algorithm 1 in :cite:`BKS11`.

    Note that this algorithm can be significantly less accurate compared to orthogonalization,
    in particular, when `V` and `W` are almost orthogonal.

    Parameters
    ----------
    V, W
        The |VectorArrays| which are to be biorthonormalized.
    product
        The inner product |Operator| w.r.t. which to biorthonormalize.
        If `None`, the Euclidean product is used.
    reiterate
        If `True`, orthonormalize again if the norm of the orthogonalized vector is
        much smaller than the norm of the original vector.
    reiteration_threshold
        If `reiterate` is `True`, re-orthonormalize if the ratio between the norms of
        the orthogonalized vector and the original vector is smaller than this value.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `V` and `W` instead of modifying `V` and `W` in-place.

    Returns
    -------
    The biorthonormalized |VectorArrays|.
    """
    assert V.space == W.space
    assert len(V) == len(W)

    logger = getLogger('pymor.algorithms.gram_schmidt.gram_schmidt_biorth')

    if copy:
        V = V.copy()
        W = W.copy()

    # main loop
    for i in range(len(V)):
        # calculate norm of V[i]
        initial_norm = V[i].norm(product)[0]

        # project V[i]
        if i == 0:
            V[0].scal(1 / initial_norm)
        else:
            norm = initial_norm
            # If reiterate is True, reiterate as long as the norm of the vector changes
            # strongly during projection.
            while True:
                for j in range(i):
                    # project by (I - V[j] * W[j]^T * E)
                    p = W[j].pairwise_inner(V[i], product)[0]
                    V[i].axpy(-p, V[j])

                # calculate new norm
                old_norm, norm = norm, V[i].norm(product)[0]

                # check if reorthogonalization should be done
                if reiterate and norm < reiteration_threshold * old_norm:
                    logger.info(f'Projecting vector V[{i}] again')
                else:
                    V[i].scal(1 / norm)
                    break

        # calculate norm of W[i]
        initial_norm = W[i].norm(product)[0]

        # project W[i]
        if i == 0:
            W[0].scal(1 / initial_norm)
        else:
            norm = initial_norm
            # If reiterate is True, reiterate as long as the norm of the vector changes
            # strongly during projection.
            while True:
                for j in range(i):
                    # project by (I - W[j] * V[j]^T * E)
                    p = V[j].pairwise_inner(W[i], product)[0]
                    W[i].axpy(-p, W[j])

                # calculate new norm
                old_norm, norm = norm, W[i].norm(product)[0]

                # check if reorthogonalization should be done
                if reiterate and norm < reiteration_threshold * old_norm:
                    logger.info(f'Projecting vector W[{i}] again')
                else:
                    W[i].scal(1 / norm)
                    break

        # rescale V[i]
        p = W[i].pairwise_inner(V[i], product)[0]
        V[i].scal(1 / p)

    if check:
        error_matrix = W.inner(V, product)
        error_matrix -= np.eye(len(V))
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError(f'result not biorthogonal (max err={err})')

    return V, W

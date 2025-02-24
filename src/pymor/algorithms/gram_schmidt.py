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


def _orth_step(A, product, i, last_iter, atol, rtol) -> tuple:
    """`cgs_iro_ls` helper function.

    Orthogonalization step for a single iteration. As long as it is not the last iteration two
    consecutive vectors are orthogonalized in one iteration, otherwise just one.
    For the last iteration case `r` (array with single value) contains the norm of the last vector
    and `proj` (array of dim [i-1,1]) the projection coefficients onto the already orthogonal basis.
    Otherwise `r` (array with two elements) contains additionally the projection between those two
    vectors and `proj` (array of dim [i-1,2]) the projection coefficients
    of the second vector onto the orthogonal basis.
    If the initial norm of the first vector is below the tolerance `atol`,
    `(None, None)` is returned to indicate that this vector can be removed and no projection
    was performed. In case projection was performed, but the first vector
    has a too low norm because of linearly dependens to the orthonormal basis,
    `(None, proj)` is returned. Otherwise `(r, proj)` is being returned.

    Parameters
    ----------
    A
        The |VectorArray| which is to be orthonormalized.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    i
        Vector in A, which has to be orthogonalized.
    last_iter
        If i is the last iteration. If so, only the last vector has to be orthogonalized.
    atol
        Vectors of norm smaller than `atol` are removed from the array.
    rtol
        Relative tolerance used to detect linear dependent vectors
        (which are then removed from the array).

    Returns
    -------
    r
        An array containing the norm of the first vector and if `last_iter` also the projection
        coefficient of the current two vectors or `None` in case the first vector can be removed.
    proj
        Array containing the projection coeffiecients of the current vector against the orthogonal
        part of A and if `last_iter` the projection coefficients of the current two vectors.
        It is of shape [i-1,1] if `last_iter` and [i-1,2] else. In case the
        first vector has a too low initial norm, None is returned.
    """
    vectors_to_orth = A[i] if last_iter else A[[i,i+1]]
    inner_result = A[:i+1].inner(vectors_to_orth, product)
    proj, r = inner_result[:-1,:], inner_result[-1,:] # split result of inner product

    initial_norm = np.sqrt(r[0])
    if initial_norm <= atol:
        return None, None

    r -= proj[:,0] @ proj.conj()

    if r[0] <= rtol * initial_norm:
        return None, proj

    r[0] = np.sqrt(r[0])

    return r, proj


@defaults('atol', 'rtol', 'check', 'check_tol')
def cgs_iro_ls(A, product=None, return_R=False, atol=1E-13, rtol=1e-13, offset=0,
               check=True, check_tol=1e-3, copy=True):
    """Orthonormalize a |VectorArray| using the CGS IRO LS algorithm.

    This method computes a QR decomposition of a |VectorArray| via the classical Gram-Schmidt
    algorithm with normalization lag and re-orthogonalization lag according to :cite:`SLAYT21`.
    Derived from the matlab repository :cite:`LCO24` to include tolerance checks,
    offsets and non-euclidean inner products.
    For condition numbers around 10^16, this algorithm starts to produce a non-orthonormal basis.

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
        Due to the instability of this algorithm it is recommendet to check for orthogonality.
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

    R = np.eye(n)

    if n == 0 or offset == n:
        return (A, R) if return_R else A

    # last orthonormal vector has to be orthogonalized again
    # to achieve an orthonormal basis
    offset = max(offset-1, 0)

    # linear dependent or vectors with a low initial norm are removed from `A`
    # (and the respective row in `R`)
    # therefore the length of A can shrink during the iterations
    # `i` is being used to keep track of the current vector
    # while `column` keeps track of the current column in `R` in which to add
    # the projection coefficients
    i = offset
    for column in range(offset, n):
        last_iter = i == len(A) - 1
        # _orth_step either returns (r, proj), (None, proj) or (None, None)
        r, proj = _orth_step(A, product, i, last_iter, atol, rtol)
        # offset being used to determine if e.g. proj contains two vectors and both can be
        # added to R or if linearly dependent components of the orthonormal basis
        # can be removed in a single function call
        lOff = 1 if r is None else 0
        rOff = 1 if last_iter else 2

        if proj is not None:
            common_dtype = np.promote_types(R.dtype, proj.dtype)
            R = R.astype(common_dtype, copy=False)
            # add proj of 2 or 1 vectors onto the orthonormal basis to R
            R[:i, column:column+rOff] += proj[:,0:rOff]
            # remove linear dependent parts of the 2 or 1 vectors
            # in case the first vector is being removed => linearly dependent parts are not removed
            A[i+lOff:i+rOff].axpy(-1.0, A[:i].lincomb(proj[:,lOff:rOff].T))

        if r is not None:
            # scale first vector
            R[i, column] = r[0]
            A[i].scal(1 / r[0])
            if not last_iter:
                R[i, column+1] = r[1] / r[0]
                # remove linear dependent parts of v1 from v2
                # i:i+1 returns an array with a single element required for lincomb
                # instead of just the element itself
                A[i+1].axpy(-1.0, A[i].lincomb(R[i:i+1, column+1]))
            i += 1
        else:
            logger.info(f'Removing vector {column}')
            del A[i]
            R = np.delete(R, [i], axis=0)

    # tolerance check
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

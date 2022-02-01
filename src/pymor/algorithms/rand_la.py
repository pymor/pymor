# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.operators.interface import Operator
from pymor.algorithms import svd_va
from pymor.algorithms.eigs import eigs
from pymor.core.logger import getLogger
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray
from pymor.operators.constructions import VectorArrayOperator, InverseOperator, IdentityOperator


@defaults('tol', 'failure_tolerance', 'num_testvecs')
def adaptive_rrf(A, source_product=None, range_product=None, tol=1e-4,
                 failure_tolerance=1e-15, num_testvecs=20, lambda_min=None, iscomplex=False):
    """Adaptive randomized range approximation of `A`.

    This is an implementation of Algorithm 1 in :cite:`BS18`.

    Given the |Operator| `A`, the return value of this method is the |VectorArray
    `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the norm denotes the
    operator norm. The inner product of the range of `A` is given by `range_product` and
    the inner product of the source of `A` is given by `source_product`.

    Parameters
    ----------
    A
        The |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    B
        |VectorArray| which contains the basis, whose span approximates the range of A.
    """
    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)
    assert isinstance(A, Operator)

    B = A.range.empty()

    R = A.source.random(num_testvecs, distribution='normal')
    if iscomplex:
        R += 1j*A.source.random(num_testvecs, distribution='normal')

    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:
        def mv(v):
            return source_product.apply(source_product.source.from_numpy(v)).to_numpy()

        def mvinv(v):
            return source_product.apply_inverse(source_product.range.from_numpy(v)).to_numpy()
        L = LinearOperator((source_product.source.dim, source_product.range.dim), matvec=mv)
        Linv = LinearOperator((source_product.range.dim, source_product.source.dim), matvec=mvinv)
        lambda_min = eigsh(L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv)[0]

    testfail = failure_tolerance / min(A.source.dim, A.range.dim)
    testlimit = np.sqrt(2. * lambda_min) * erfinv(testfail**(1. / num_testvecs)) * tol
    maxnorm = np.inf
    M = A.apply(R)

    while(maxnorm > testlimit):
        basis_length = len(B)
        v = A.source.random(distribution='normal')
        if iscomplex:
            v += 1j*A.source.random(distribution='normal')
        B.append(A.apply(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))

    return B


@defaults('q', 'l')
def rrf(A, source_product=None, range_product=None, q=2, l=8, return_rand=False, iscomplex=False):
    """Randomized range approximation of `A`.

    This is an implementation of Algorithm 4.4 in :cite:`HMT11`.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `Q` whose vectors form an approximate orthonomal basis for the range of `A`.

    Parameters
    ----------
    A
        The |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    q
        The number of power iterations.
    l
        The block size of the normalized power iterations.
    return_rand
        If 'True', the randomly sampled |VectorArray| R is returned. 
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    Q
        |VectorArray| which contains the basis, whose span approximates the range of A.
    R 
        The randomly sampled |VectorArray|
    """
     assert isinstance(A, Operator)
    
    if range_product is None:
        range_product = IdentityOperator(A.range)
    else:
        assert isinstance(range_product, Operator)
        
    if source_product is None:
        source_product = IdentityOperator(A.source)
    else:
        assert isinstance(source_product, Operator)
    
    R = A.source.random(l, distribution='normal')

    if iscomplex:
        R += 1j*A.source.random(l, distribution='normal')

    Q = A.apply(R)
    gram_schmidt(Q, range_product, atol=0, rtol=0, copy=False)

    for i in range(q):
        Q = A.apply_adjoint(range_product.apply(Q))
        Q = source_product.apply_inverse(Q)
        gram_schmidt(Q, source_product, atol=0, rtol=0, copy=False)
        Q = A.apply(Q)
        gram_schmidt(Q, range_product, atol=0, rtol=0, copy=False)

    if return_rand:
        return Q, R
    else:
        return Q


    @defaults('p', 'q')
    def random_generalized_svd(A, S=None, T=None, modes=None, p=20, q=2):
    
    """ Randomized SVD of a |VectorArray|. 
    
    Viewing the |VectorArray| 'A' as a 'A.dim' x 'len(A)' matrix, the return value
    of this method is the randomized singular value decomposition of 'A', where the
    inner product on R^('dim(A)') is given by 'S' and the inner product on R^('len(A)')
    is given by 'T'. 
        
        A = U*\Sigma*V*T
    
    Parameters
    ----------
    A : 
        The |VectorArray| or |Operator| for which the randomized SVD is to be computed. 
    S : 
        Range product |Operator| w.r.t which the randomized SVD is computed

    T :
        Source product |Operator| w.r.t which the randomized SVD is computed

    modes : 
        If not 'None', at most the first 'modes' approximated singular values 
        and vectors are returned. 
    p :
        If not '0', adds 'Oversampling' colums to the randomly sampled matrix 'G'. 
    q : 
        If not '0', performs 'PowerIterations' iterations to increase the relative weight
        of the first singular values.

    Returns
    -------
    U 
        |VectorArray| of approximated left singular vectors 
    s 
        One-dimensional |NumPy array| of the approximated singular values
    V
        |VectorArray| of the approximated right singular vectors
    """
    logger = getLogger('pymor.algorithms.rand_la')
    
    assert isinstance(A, VectorArray) or isinstance(A, Operator)
    
    if isinstance(A, VectorArray):
        A = VectorArrayOperator(A)
   
    assert 0 <= modes <= max(A.source.dim, A.range.dim) and isinstance(modes,int)
    assert p >= 0 and isinstance(p, int)
    assert q >= 0 and isinstance(q, int)
    
    if S is None:
        S = IdentityOperator(A.range)
    else:
        assert isinstance(S, Operator)
        
    if T is None:
        T = IdentityOperator(A.source)
    else:
        assert isinstance(T, Operator)
        
    if A.source.dim == 0 or A.range.dim == 0:
        return A.source.empty(), np.array([]), A.range.empty()
    
    Q = rrf(A, source_product=T, range_product=S, q=q, l=modes+p)
    B = A.apply_adjoint(S.apply(Q))
    Q_B, R_B = gram_schmidt(T.apply_inverse(B), product=T, return_R=True)
    U_b, s, V_b = sp.linalg.svd(R_B.T)

    with logger.block(f'Computing generalized left-singular vectors ({U_b[:,:modes].shape[1]} vectors) ...'):
        U = Q.lincomb(U_b)

    with logger.block(f'Computung gerneralized right-singular vector ({V_b[:modes,:].shape[0]} vectors) ...'):
        V = Q_B.lincomb(V_b.T)
        
    return  U[:modes], s[:modes], V[:modes]


    @defaults('p', 'q','which','maxIter','complex_evp','left_evp','seed')
    def random_ghep(A, E=None, k=None, p=20, q=2, sigma=None, which='LM', b=None, l=None, maxIter=1000, complex_evp=False, left_evp=False, seed=0):

     """Approximates a few eigenvalues of a linear |Operator| with randomized methods.

    Approximates `k` eigenvalues `w` with corresponding eigenvectors `v` which solve
    the eigenvalue problem

    .. math::
        A v_i = w_i v_i

    or the generalized eigenvalue problem

    .. math::
        A v_i = w_i E v_i

    if `E` is not `None`.

    Parameters
    ----------
    A
        The linear |Operator| for which the eigenvalues are to be computed.
    E
        The linear |Operator| which defines the generalized eigenvalue problem.
    k
        The number of eigenvalues and eigenvectors which are to be computed.
    p :
        If not '0', adds 'p' colums to the randomly sampled matrix in the randrangefinder.rff() method.
        Often called Oversampling parameter.
    q : 
        If not '0', performs 'q' PowerIterations to increase the relative weight
        of the bigger singular values.
    sigma
        If not `None` transforms the eigenvalue problem such that the k eigenvalues
        closest to sigma are computed.
    which
        A string specifying which `k` eigenvalues and eigenvectors to compute:

        - `'LM'`: select eigenvalues with largest magnitude
        - `'SM'`: select eigenvalues with smallest magnitude
        - `'LR'`: select eigenvalues with largest real part
        - `'SR'`: select eigenvalues with smallest real part
        - `'LI'`: select eigenvalues with largest imaginary part
        - `'SI'`: select eigenvalues with smallest imaginary part
    b
        Initial vector for Arnoldi iteration. Default is a random vector.
    l
        The size of the Arnoldi factorization. Default is `min(n - 1, max(2*k + 1, 20))`.
    maxiter
        The maximum number of iterations.
    complex_evp
        Wether to consider an eigenvalue problem with complex operators. When operators
        are real setting this argument to `False` will increase stability and performance.
    left_evp
        If set to `True` compute left eigenvectors else compute right eigenvectors.
    seed
        Random seed which is used for computing the initial vector for the Arnoldi
        iteration.

    Returns
    -------
    w
        A 1D |NumPy array| which contains the computed eigenvalues.
    V
        A |VectorArray| which contains the computed eigenvectors.
    """
    logger = getLogger('pymor.algorithms.rand_la')
    
    assert isinstance(A, Operator) and A.linear
    assert not A.parametric
    assert A.source == A.range
    assert 0 <= modes <= max(A.source.dim, A.range.dim) and isinstance(modes, int)
    assert p >= 0 and isinstance(p, int)
    assert q >= 0 and isinstance(q, int)
    
    if E is None:
        E = IdentityOperator(A.source)
    else:
        assert isinstance(E, Operator) and E.linear
        assert not E.parametric
        assert E.source == E.range
        assert E.source == A.source
    
    if A.source.dim == 0 or A.range.dim == 0:
        return A.source.empty(), np.array([]), A.range.empty()
    
    C = InverseOperator(E) @ A
    Y, Omega = rrf(C, q=q, l=modes+p, return_rand=True)
    Omega_op = VectorArrayOperator(Omega)
    Q = gram_schmidt(Y, product=E)
    Q_op = VectorArrayOperator(Q)
    
    if single_pass :
        X = VectorArrayOperator(Omega_op.apply_adjoint(E.apply(Q)))
        Z = VectorArrayOperator(Q_op.apply_adjoint(E.apply(Omega)))
        T = InverseOperator(X) @ VectorArrayOperator(Omega_op.apply_adjoint(A.apply(Q))) @ InverseOperator(Z)
    else: 
        T = Q_op.H @ A @ Q_op
        
    w, S = eigs(T,k=modes)
    
    with logger.block(f'Computing eigenvectors ({S.dim} vectors) ...'):
        V = Q_op.apply(S)
        
    return w, V
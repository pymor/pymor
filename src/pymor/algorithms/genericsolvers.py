# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module contains some iterative linear solvers which only use the |Operator| interface"""

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError
from pymor.core.logger import getLogger


@defaults('lgmres_tol', 'lgmres_maxiter',
          'lgmres_inner_m', 'lgmres_outer_k', 'least_squares_lsmr_damp',
          'least_squares_lsmr_atol', 'least_squares_lsmr_btol', 'least_squares_lsmr_conlim',
          'least_squares_lsmr_maxiter', 'least_squares_lsmr_show',
          'least_squares_lsqr_atol', 'least_squares_lsqr_btol', 'least_squares_lsqr_conlim',
          'least_squares_lsqr_iter_lim', 'least_squares_lsqr_show')
def solver_options(lgmres_tol=1e-5,
                   lgmres_maxiter=1000,
                   lgmres_inner_m=39,
                   lgmres_outer_k=3,
                   least_squares_lsmr_damp=0.0,
                   least_squares_lsmr_atol=1e-6,
                   least_squares_lsmr_btol=1e-6,
                   least_squares_lsmr_conlim=1e8,
                   least_squares_lsmr_maxiter=None,
                   least_squares_lsmr_show=False,
                   least_squares_lsqr_damp=0.0,
                   least_squares_lsqr_atol=1e-6,
                   least_squares_lsqr_btol=1e-6,
                   least_squares_lsqr_conlim=1e8,
                   least_squares_lsqr_iter_lim=None,
                   least_squares_lsqr_show=False):
    """Returns available solvers with default |solver_options|.

    Parameters
    ----------
    lgmres_tol
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_maxiter
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_inner_m
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_outer_k
        See :func:`scipy.sparse.linalg.lgmres`.
    least_squares_lsmr_damp
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_atol
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_btol
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_conlim
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_maxiter
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_show
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsqr_damp
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_atol
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_btol
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_conlim
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_iter_lim
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_show
        See :func:`scipy.sparse.linalg.lsqr`.

    Returns
    -------
    A dict of available solvers with default |solver_options|.
    """
    return {'generic_lgmres': {'type': 'generic_lgmres',
                               'tol': lgmres_tol,
                               'maxiter': lgmres_maxiter,
                               'inner_m': lgmres_inner_m,
                               'outer_k': lgmres_outer_k},
            'generic_least_squares_lsmr': {'type': 'generic_least_squares_lsmr',
                                           'damp': least_squares_lsmr_damp,
                                           'atol': least_squares_lsmr_atol,
                                           'btol': least_squares_lsmr_btol,
                                           'conlim': least_squares_lsmr_conlim,
                                           'maxiter': least_squares_lsmr_maxiter,
                                           'show': least_squares_lsmr_show},
            'generic_least_squares_lsqr': {'type': 'generic_least_squares_lsqr',
                                           'damp': least_squares_lsqr_damp,
                                           'atol': least_squares_lsqr_atol,
                                           'btol': least_squares_lsqr_btol,
                                           'conlim': least_squares_lsqr_conlim,
                                           'iter_lim': least_squares_lsqr_iter_lim,
                                           'show': least_squares_lsqr_show}}


@defaults('check_finite', 'default_solver', 'default_least_squares_solver')
def apply_inverse(op, V, initial_guess=None, options=None, least_squares=False, check_finite=True,
                  default_solver='generic_lgmres', default_least_squares_solver='generic_least_squares_lsmr'):
    """Solve linear equation system.

    Applies the inverse of `op` to the vectors in `V` using a generic iterative solver.

    Parameters
    ----------
    op
        The linear, non-parametric |Operator| to invert.
    V
        |VectorArray| of right-hand sides for the equation system.
    initial_guess
        |VectorArray| with the same length as `V` containing initial guesses
        for the solution.  Some implementations of `apply_inverse` may
        ignore this parameter.  If `None` a solver-dependent default is used.
    options
        The |solver_options| to use (see :func:`solver_options`).
    least_squares
        If `True`, return least squares solution.
    check_finite
        Test if solution only contains finite values.
    default_solver
        Default solver to use (generic_lgmres, generic_least_squares_lsmr,
        generic_least_squares_lsqr).
    default_least_squares_solver
        Default solver to use for least squares problems (generic_least_squares_lsmr,
        generic_least_squares_lsqr).

    Returns
    -------
    |VectorArray| of the solution vectors.
    """
    assert V in op.range
    assert initial_guess is None or initial_guess in op.source and len(initial_guess) == len(V)
    options = _parse_options(options, solver_options(), default_solver, default_least_squares_solver, least_squares)

    R = op.source.empty(reserve=len(V))

    if options['type'] == 'generic_lgmres':
        for i in range(len(V)):
            r, info = lgmres(op, V[i], initial_guess[i] if initial_guess is not None else None,
                             tol=options['tol'],
                             maxiter=options['maxiter'],
                             inner_m=options['inner_m'],
                             outer_k=options['outer_k'])
            if info > 0:
                raise InversionError(f'lgmres failed to converge after {info} iterations')
            assert info == 0
            R.append(r)
    elif options['type'] == 'generic_least_squares_lsmr':
        for i in range(len(V)):
            r, info, itn, _, _, _, _, _ = lsmr(op, V[i],
                                               damp=options['damp'],
                                               atol=options['atol'],
                                               btol=options['btol'],
                                               conlim=options['conlim'],
                                               maxiter=options['maxiter'],
                                               show=options['show'])
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError(f'lsmr failed to converge after {itn} iterations')
            getLogger('pymor.algorithms.genericsolvers.lsmr').info(f'Converged after {itn} iterations')
            R.append(r)
    elif options['type'] == 'generic_least_squares_lsqr':
        for i in range(len(V)):
            r, info, itn, _, _, _, _, _, _ = lsqr(op, V[i],
                                                  damp=options['damp'],
                                                  atol=options['atol'],
                                                  btol=options['btol'],
                                                  conlim=options['conlim'],
                                                  iter_lim=options['iter_lim'],
                                                  show=options['show'])
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError(f'lsmr failed to converge after {itn} iterations')
            getLogger('pymor.algorithms.genericsolvers.lsqr').info(f'Converged after {itn} iterations')
            R.append(r)
    else:
        raise ValueError('Unknown solver type')

    if check_finite:
        if not np.isfinite(np.all(R.norm())):
            raise InversionError('Result contains non-finite values')

    return R


def _parse_options(options, default_options, default_solver, default_least_squares_solver, least_squares):
    if options is None:
        options = default_options[default_least_squares_solver] if least_squares else default_options[default_solver]
    elif isinstance(options, str):
        options = default_options[options]
    else:
        assert 'type' in options and options['type'] in default_options \
            and options.keys() <= default_options[options['type']].keys()
        user_options = options
        options = default_options[user_options['type']]
        options.update(user_options)

    if least_squares != ('least_squares' in options['type']):
        logger = getLogger('foo')
        if least_squares:
            logger.warning('Non-least squares solver selected for least squares problem.')
        else:
            logger.warning('Least squares solver selected for non-least squares problem.')

    return options


# The following code is an adapted version of
# scipy.sparse.linalg.lgmres.
# Original copyright notice:
#
# Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as Scipy.


def lgmres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
           inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True):
    if A.source != A.range:
        raise InversionError
    from scipy.linalg import lstsq
    x = A.source.zeros() if x0 is None else x0.copy()

    # psolve = M.matvec

    if outer_v is None:
        outer_v = []

    b_norm = b.norm()[0]
    if b_norm == 0:
        b_norm = 1

    for k_outer in range(maxiter):
        r_outer = A.apply(x) - b

        # -- callback
        if callback is not None:
            callback(x)

        # -- check stopping condition
        r_norm = r_outer.norm()[0]
        if r_norm < tol * b_norm or r_norm < tol:
            break

        # -- inner LGMRES iteration
        vs0 = -r_outer   # -psolve(r_outer)
        inner_res_0 = vs0.norm()[0]

        if inner_res_0 == 0:
            rnorm = r_outer.norm()[0]
            raise RuntimeError("Preconditioner returned a zero vector; "
                               "|v| ~ %.1g, |M v| = 0" % rnorm)

        vs0.scal(1.0/inner_res_0)
        hs = []
        vs = [vs0]
        ws = []
        y = None

        for j in range(1, 1 + inner_m + len(outer_v)):
            # -- Arnoldi process:
            #
            #    Build an orthonormal basis V and matrices W and H such that
            #        A W = V H
            #    Columns of W, V, and H are stored in `ws`, `vs` and `hs`.
            #
            #    The first column of V is always the residual vector, `vs0`;
            #    V has *one more column* than the other of the three matrices.
            #
            #    The other columns in V are built by feeding in, one
            #    by one, some vectors `z` and orthonormalizing them
            #    against the basis so far. The trick here is to
            #    feed in first some augmentation vectors, before
            #    starting to construct the Krylov basis on `v0`.
            #
            #    It was shown in :cite:`BJM` that a good choice (the LGMRES choice)
            #    for these augmentation vectors are the `dx` vectors obtained
            #    from a couple of the previous restart cycles.
            #
            #    Note especially that while `vs0` is always the first
            #    column in V, there is no reason why it should also be
            #    the first column in W. (In fact, below `vs0` comes in
            #    W only after the augmentation vectors.)
            #
            #    The rest of the algorithm then goes as in GMRES, one
            #    solves a minimization problem in the smaller subspace
            #    spanned by W (range) and V (image).
            #
            #    XXX: Below, I'm lazy and use `lstsq` to solve the
            #    small least squares problem. Performance-wise, this
            #    is in practice acceptable, but it could be nice to do
            #    it on the fly with Givens etc.
            #

            #     ++ evaluate
            v_new = None
            if j < len(outer_v) + 1:
                z, v_new = outer_v[j-1]
            elif j == len(outer_v) + 1:
                z = vs0
            else:
                z = vs[-1]

            if v_new is None:
                v_new = A.apply(z)  # psolve(matvec(z))
            else:
                # Note: v_new is modified in-place below. Must make a
                # copy to ensure that the outer_v vectors are not
                # clobbered.
                v_new = v_new.copy()

            #     ++ orthogonalize
            hcur = []
            for v in vs:
                alpha = v.inner(v_new)[0, 0]
                hcur.append(alpha)
                v_new.axpy(-alpha, v)  # v_new -= alpha*v
            hcur.append(v_new.norm()[0])

            if hcur[-1] == 0:
                # Exact solution found; bail out.
                # Zero basis vector (v_new) in the least-squares problem
                # does no harm, so we can just use the same code as usually;
                # it will give zero (inner) residual as a result.
                bailout = True
            else:
                bailout = False
                v_new.scal(1.0/hcur[-1])

            vs.append(v_new)
            hs.append(hcur)
            ws.append(z)

            # XXX: Ugly: should implement the GMRES iteration properly,
            #      with Givens rotations and not using lstsq. Instead, we
            #      spare some work by solving the LSQ problem only every 5
            #      iterations.
            if not bailout and j % 5 != 1 and j < inner_m + len(outer_v) - 1:
                continue

            # -- GMRES optimization problem
            hess = np.zeros((j+1, j))
            e1 = np.zeros((j+1,))
            e1[0] = inner_res_0
            for q in range(j):
                hsq = np.array(hs[q])
                common_dtype = np.promote_types(hess.dtype, hsq.dtype)
                hess = hess.astype(common_dtype, copy=False)
                hess[:(q+2), q] = hsq

            y, resids, rank, s = lstsq(hess, e1)
            inner_res = np.linalg.norm(np.dot(hess, y) - e1)

            # -- check for termination
            if inner_res < tol * inner_res_0:
                break

        # -- GMRES terminated: eval solution
        dx = ws[0]*y[0]
        for w, yc in zip(ws[1:], y[1:]):
            dx.axpy(yc, w)  # dx += w*yc

        # -- Store LGMRES augmentation vectors
        nx = dx.norm()[0]
        if store_outer_Av:
            q = np.dot(hess, y)
            ax = vs[0]*q[0]
            for v, qc in zip(vs[1:], q[1:]):
                ax.axpy(qc, v)
            outer_v.append((dx * (1./nx), ax * (1./nx)))
        else:
            outer_v.append((dx * (1./nx), None))

        # -- Retain only a finite number of augmentation vectors
        while len(outer_v) > outer_k:
            del outer_v[0]

        # -- Apply step
        x += dx
    else:
        # didn't converge ...
        return x, maxiter

    getLogger('pymor.algorithms.genericsolvers.lgmres').info(f'Converged after {k_outer+1} iterations')

    return x, 0


# The following code is an adapted version of
# scipy.sparse.linalg.lsqr.
# Original copyright notice:
#
# Sparse Equations and Least Squares.
#
# The original Fortran code was written by C. C. Paige and M. A. Saunders as
# described in
#
# C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
# equations and sparse least squares, TOMS 8(1), 43--71 (1982).
#
# C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
# equations and least-squares problems, TOMS 8(2), 195--209 (1982).
#
# It is licensed under the following BSD license:
#
# Copyright (c) 2006, Systems Optimization Laboratory
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Stanford University nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The Fortran code was translated to Python for use in CVXOPT by Jeffery
# Kline with contributions by Mridul Aanjaneya and Bob Myhill.
#
# Adapted for SciPy by Stefan van der Walt.


def _sym_ortho(a, b):
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / np.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / np.sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r


def lsqr(A, b, damp=0.0, atol=1e-8, btol=1e-8, conlim=1e8,
         iter_lim=None, show=False):
    m, n = A.range.dim, A.source.dim
    if iter_lim is None:
        iter_lim = 2 * n

    msg = ('The exact solution is  x = 0                              ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = f'The matrix A has {m:8g} rows  and {n:8g} cols'
        str2 = 'damp = %20.14e  ' % (damp)
        str3 = f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}'
        str4 = f'btol = {btol:8.2e}               iter_lim = {iter_lim:8g}'
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn = 0
    istop = 0
    # nstop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    """
    Set up the first vectors u and v for the bidiagonalization.
    These satisfy  beta*u = b,  alfa*v = A'u.
    """
    # __xm = A.range.zeros()  # a matrix for temporary holding
    # __xn = A.source.zeros()  # a matrix for temporary holding
    v = A.source.zeros()
    u = b.copy()
    x = A.source.zeros()
    alfa = 0
    beta = u.norm()[0]
    w = A.source.zeros()

    if beta > 0:
        u.scal(1/beta)
        v = A.apply_adjoint(u)
        alfa = v.norm()[0]

    if alfa > 0:
        v.scal(1/alfa)
        w = v.copy()

    rhobar = alfa
    phibar = beta
    bnorm = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        print(msg[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm

    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'

    if show:
        print(' ')
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = f'{itn:6g} {x.dofs([0])[0]:12.5e}'
        str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(str1, str2, str3)

    # Main iteration loop.
    while itn < iter_lim:
        itn = itn + 1
        """
        %     Perform the next step of the bidiagonalization to obtain the
        %     next  beta, u, alfa, v.  These satisfy the relations
        %                beta*u  =  a*v   -  alfa*u,
        %                alfa*v  =  A'*u  -  beta*v.
        """
        u = A.apply(v) - u * alfa
        beta = u.norm()[0]

        if beta > 0:
            u.scal(1/beta)
            anorm = np.sqrt(anorm**2 + alfa**2 + beta**2 + damp**2)
            v = A.apply_adjoint(u) - v * beta
            alfa = v.norm()[0]
            if alfa > 0:
                v.scal(1 / alfa)

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        rhobar1 = np.sqrt(rhobar**2 + damp**2)
        cs1 = rhobar / rhobar1
        sn1 = damp / rhobar1
        psi = sn1 * phibar
        phibar = cs1 * phibar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = w * (1 / rho)

        x = x + w * t1
        w = v + w * t2
        ddnorm = ddnorm + dk.norm2()[0]

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = np.sqrt(xxnorm + zbar**2)
        gamma = np.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * np.sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = np.sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq = rnorm**2 - dampsq * xxnorm
        r1norm = np.sqrt(abs(r1sq))
        if r1sq < 0:
            r1norm = -r1norm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm)
        test3 = 1 / acond
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.
        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 10:
            prnt = True
        if itn >= iter_lim-10:
            prnt = True
        # if itn%10 == 0: prnt = True
        if test3 <= 2*ctol:
            prnt = True
        if test2 <= 10*atol:
            prnt = True
        if test1 <= 10*rtol:
            prnt = True
        if istop != 0:
            prnt = True

        if prnt:
            if show:
                str1 = f'{itn:6g} {x.dofs([0])[0]:12.5e}'
                str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
                str3 = f'  {test1:8.1e} {test2:8.1e}'
                str4 = f' {anorm:8.1e} {acond:8.1e}'
                print(str1, str2, str3, str4)

        if istop != 0:
            break

    # End of iteration loop.
    # Print the stopping condition.
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        str1 = f'istop ={istop:8g}   r1norm ={r1norm:8.1e}'
        str2 = f'anorm ={anorm:8.1e}   arnorm ={arnorm:8.1e}'
        str3 = f'itn   ={itn:8g}   r2norm ={r2norm:8.1e}'
        str4 = f'acond ={acond:8.1e}   xnorm  ={xnorm:8.1e}'
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm


# The following code is an adapted version of
# scipy.sparse.linalg.lsqr.
# Original copyright notice:
#
# Copyright (C) 2010 David Fong and Michael Saunders
#
# LSMR uses an iterative method.
#
# 07 Jun 2010: Documentation updated
# 03 Jun 2010: First release version in Python
#
# David Chin-lung Fong            clfong@stanford.edu
# Institute for Computational and Mathematical Engineering
# Stanford University
#
# Michael Saunders                saunders@stanford.edu
# Systems Optimization Laboratory
# Dept of MS&E, Stanford University.


def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None, show=False):

    msg = ('The exact solution is  x = 0                              ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    hdg1 = '   itn      x(1)       norm r    norm A''r'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20   # print frequency (for repeating the heading)
    pcount = 0   # print counter

    m, n = A.range.dim, A.source.dim

    # stores the num of singular values
    minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim

    if show:
        print(' ')
        print('LSMR            Least-squares solution of  Ax = b\n')
        print(f'The matrix A has {m:8g} rows  and {n:8g} cols')
        print('damp = %20.14e\n' % (damp))
        print(f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}\n')
        print(f'btol = {btol:8.2e}             maxiter = {maxiter:8g}\n')

    u = b.copy()
    beta = u.norm()[0]

    v = A.source.zeros()
    alpha = 0

    if beta > 0:
        u.scal(1 / beta)
        v = A.apply_adjoint(u)
        alpha = v.norm()[0]

    if alpha > 0:
        v.scal(1 / alpha)

    # Initialize variables for 1st iteration.

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = A.source.zeros()
    x = A.source.zeros()

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = np.sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules.
    normb = beta
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        if show:
            print(msg[0])
        return x, istop, itn, normr, normar, normA, condA, normx

    if show:
        print(' ')
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = f'{itn:6g} {x.dofs([0])[0]:12.5e}'
        str2 = f' {normr:10.3e} {normar:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(''.join([str1, str2, str3]))

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.

        u = A.apply(v) - u * alpha
        beta = u.norm()[0]

        if beta > 0:
            u.scal(1 / beta)
            v = A.apply_adjoint(u) - v * beta
            alpha = v.norm()[0]
            if alpha > 0:
                v.scal(1 / alpha)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.

        hbar = h - hbar * (thetabar * rho / (rhoold * rhobarold))
        x = x + hbar * (zeta / (rho * rhobar))
        h = v - h * (thetanew / rho)

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = np.sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = np.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = x.norm()[0]

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = np.infty
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.

        if show:
            if (n <= 40) or (itn <= 10) or (itn >= maxiter - 10) or \
               (itn % 10 == 0) or (test3 <= 1.1 * ctol) or \
               (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or \
               (istop != 0):

                if pcount >= pfreq:
                    pcount = 0
                    print(' ')
                    print(hdg1, hdg2)
                pcount = pcount + 1
                str1 = f'{itn:6g} {x.dofs([0])[0]:12.5e}'
                str2 = f' {normr:10.3e} {normar:10.3e}'
                str3 = f'  {test1:8.1e} {test2:8.1e}'
                str4 = f' {normA:8.1e} {condA:8.1e}'
                print(''.join([str1, str2, str3, str4]))

        if istop > 0:
            break

    # Print the stopping condition.

    if show:
        print(' ')
        print('LSMR finished')
        print(msg[istop])
        print(f'istop ={istop:8g}    normr ={normr:8.1e}')
        print(f'    normA ={normA:8.1e}    normAr ={normar:8.1e}')
        print(f'itn   ={itn:8g}    condA ={condA:8.1e}')
        print('    normx =%8.1e' % (normx))
        print(str1, str2)
        print(str3, str4)

    return x, istop, itn, normr, normar, normA, condA, normx

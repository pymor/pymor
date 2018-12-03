# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_SLYCOT:
    import numpy as np
    import slycot

    from pymor.algorithms.genericsolvers import _parse_options
    from pymor.algorithms.lyapunov import _solve_lyap_check_args, chol
    from pymor.algorithms.to_matrix import to_matrix
    from pymor.bindings.scipy import _solve_ricc_check_args
    from pymor.core.logger import getLogger

    def _solve_check(dtype, solver, sep, ferr):
        if ferr > np.sqrt(np.finfo(dtype).eps):
            logger = getLogger(solver)
            logger.warning('Estimated forward relative error bound is large (ferr={:e}, sep={:e}). '
                           'Result may not be accurate.'.format(ferr, sep))

    def lyap_lrcf_solver_options():
        """Returns available Lyapunov equation solvers with default solver options for the slycot backend.

        Returns
        -------
        A dict of available solvers with default solver options.
        """

        return {'slycot_bartels-stewart': {'type': 'slycot_bartels-stewart'}}

    def solve_lyap_lrcf(A, E, B, trans=False, options=None):
        """Compute an approximate low-rank solution of a Lyapunov equation.

        See :func:`pymor.algorithms.lyapunov.solve_lyap_lrcf` for a
        general description.

        This function uses `slycot.sb03md` (if `E is None`) and
        `slycot.sg03ad` (if `E is not None`), which are dense solvers
        based on the Bartels-Stewart algorithm.
        Therefore, we assume A, E, and B can be converted to |NumPy
        arrays| using :func:`~pymor.algorithms.to_matrix.to_matrix`.

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        trans
            Whether the first |Operator| in the Lyapunov equation is
            transposed.
        options
            The solver options to use (see
            :func:`lyap_lrcf_solver_options`).

        Returns
        -------
        Z
            Low-rank Cholesky factor of the Lyapunov equation solution,
            |VectorArray| from `A.source`.
        """

        _solve_lyap_check_args(A, E, B, trans)
        options = _parse_options(options, lyap_lrcf_solver_options(), 'slycot_bartels-stewart', None, False)

        if options['type'] == 'slycot_bartels-stewart':
            X = solve_lyap_dense(A, E, B, trans=trans)
            Z = chol(X)
        else:
            raise ValueError('Unexpected Lyapunov equation solver ({}).'.format(options['type']))
        Z = A.source.from_numpy(np.array(Z).T)
        return Z

    def lyap_dense_solver_options():
        """Returns available Lyapunov equation solvers with default solver options for the Slycot backend.

        Returns
        -------
        A dict of available solvers with default solver options.
        """

        return {'slycot_bartels-stewart': {'type': 'slycot_bartels-stewart'}}

    def solve_lyap_dense(A, E, B, trans=False, options=None):
        """Compute the solution of a Lyapunov equation.

        See :func:`pymor.algorithms.lyapunov.solve_lyap_dense` for a
        general description.

        This function uses `slycot.sb03md` (if `E is None`) and
        `slycot.sg03ad` (if `E is not None`), which are based on the
        Bartels-Stewart algorithm.

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        trans
            Whether the first |Operator| in the Lyapunov equation is
            transposed.
        options
            The solver options to use (see
            :func:`lyap_dense_solver_options`).

        Returns
        -------
        X
            Lyapunov equation solution as a |NumPy array|.
        """

        _solve_lyap_check_args(A, E, B, trans)
        options = _parse_options(options, lyap_dense_solver_options(), 'slycot_bartels-stewart', None, False)

        if options['type'] == 'slycot_bartels-stewart':
            A = to_matrix(A, format='dense')
            n = A.shape[0]
            if E is not None:
                E = to_matrix(E, format='dense')
            B = to_matrix(B, format='dense')
            if not trans:
                C = -B.dot(B.T)
                trana = 'T'
            else:
                C = -B.T.dot(B)
                trana = 'N'
            dico = 'C'
            job = 'B'
            if E is None:
                U = np.zeros((n, n))
                X, scale, sep, ferr, _ = slycot.sb03md(n, C, A, U, dico, job=job, trana=trana)
                _solve_check(A.dtype, 'slycot.sb03md', sep, ferr)
            else:
                fact = 'N'
                uplo = 'L'
                Q = np.zeros((n, n))
                Z = np.zeros((n, n))
                _, _, _, _, X, scale, sep, ferr, _, _, _ = slycot.sg03ad(dico, job, fact, trana, uplo,
                                                                         n, A, E,
                                                                         Q, Z, C)
                _solve_check(A.dtype, 'slycot.sg03ad', sep, ferr)
            X /= scale
        else:
            raise ValueError('Unexpected Lyapunov equation solver ({}).'.format(options['type']))

        return X

    def ricc_lrcf_solver_options():
        """Returns available Riccati equation solvers with default solver options for the SciPy backend.

        Returns
        -------
        A dict of available solvers with default solver options.
        """

        return {'slycot': {'type': 'slycot'}}

    def solve_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
        """Compute an approximate low-rank solution of a Riccati equation.

        See :func:`pymor.algorithms.riccati.solve_ricc_lrcf` for a
        general description.

        This function uses `slycot.sb02md` (if E and S are `None`),
        `slycot.sb02od` (if E is `None` and S is not `None`) and
        `slycot.sg03ad` (if E is not `None`), which are dense solvers.
        Therefore, we assume all |Operators| can be converted to |NumPy
        arrays| using :func:`~pymor.algorithms.to_matrix.to_matrix`.

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        C
            The |Operator| C.
        R
            The |Operator| R or `None`.
        S
            The |Operator| S or `None`.
        trans
            Whether the first |Operator| in the Riccati equation is
            transposed.
        options
            The solver options to use (see
            :func:`ricc_lrcf_solver_options`).

        Returns
        -------
        Z
            Low-rank Cholesky factor of the Riccati equation solution,
            |VectorArray| from `A.source`.
        """

        _solve_ricc_check_args(A, E, B, C, R, S, trans)
        options = _parse_options(options, ricc_lrcf_solver_options(), 'slycot', None, False)
        if options['type'] != 'slycot':
            raise ValueError('Unexpected Riccati equation solver ({}).'.format(options['type']))

        A_source = A.source
        A = to_matrix(A, format='dense')
        E = to_matrix(E, format='dense') if E else None
        B = to_matrix(B, format='dense') if B else None
        C = to_matrix(C, format='dense') if C else None
        R = to_matrix(R, format='dense') if R else None
        S = to_matrix(S, format='dense') if S else None

        n = A.shape[0]
        dico = 'C'

        if E is None:
            if S is None:
                if not trans:
                    A = A.T
                    if R is None:
                        G = C.T.dot(C)
                    else:
                        G = slycot.sb02mt(n, C.shape[0], C.T, R)[-1]
                    Q = B.dot(B.T)
                else:
                    if R is None:
                        G = B.dot(B.T)
                    else:
                        G = slycot.sb02mt(n, B.shape[1], B, R)[-1]
                    Q = C.T.dot(C)
                X, rcond = slycot.sb02md(n, A, G, Q, dico)[:2]
                _ricc_rcond_check('slycot.sb02md', rcond)
            else:
                if not trans:
                    m = C.shape[0]
                    p = B.shape[1]
                    if R is None:
                        R = np.eye(m)
                    X, rcond = slycot.sb02od(n, m, A.T, C.T, B.T, R, dico, p=p, L=S, fact='C')[:2]
                else:
                    m = B.shape[1]
                    p = C.shape[0]
                    if R is None:
                        R = np.eye(m)
                    X, rcond = slycot.sb02od(n, m, A, B, C, R, dico, p=p, L=S, fact='C')[:2]
                _ricc_rcond_check('slycot.sb02od', rcond)
        else:
            jobb = 'B'
            fact = 'C'
            uplo = 'U'
            jobl = 'Z' if S is None else 'N'
            scal = 'N'
            sort = 'S'
            acc = 'R'
            if not trans:
                m = C.shape[0]
                p = B.shape[1]
                if R is None:
                    R = np.eye(m)
                if S is None:
                    S = np.empty((n, m))
                out = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc,
                                    n, m, p,
                                    A.T, E.T, C.T, B.T, R, S)
            else:
                m = B.shape[1]
                p = C.shape[0]
                if R is None:
                    R = np.eye(m)
                if S is None:
                    S = np.empty((n, m))
                out = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc,
                                    n, m, p,
                                    A, E, B, C, R, S)
            rcond = out[0]
            X = out[1]
            _ricc_rcond_check('slycot.sg02ad', rcond)

        Z = chol(X)
        Z = A_source.from_numpy(np.array(Z).T)
        return Z


def _ricc_rcond_check(solver, rcond):
    if rcond < np.sqrt(np.finfo(np.float64).eps):
        logger = getLogger(solver)
        logger.warning('Estimated reciprocal condition number is small (rcond={:e}). '
                       'Result may not be accurate.'.format(rcond))

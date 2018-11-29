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

    def ricc_solver_options():
        """Returns available Riccati equation solvers with default |solver_options| for the SciPy backend.

        Returns
        -------
        A dict of available solvers with default |solver_options|.
        """

        return {'slycot': {'type': 'slycot'}}

    def solve_ricc(A, E=None, B=None, Q=None, C=None, R=None, G=None, trans=False, options=None):
        """Find a factor of the solution of a Riccati equation

        Returns factor :math:`Z` such that :math:`Z Z^T` is
        approximately the solution :math:`X` of a Riccati equation

        .. math::
            A^T X E + E^T X A - E^T X B R^{-1} B^T X E + Q = 0.

        If E in `None`, it is taken to be the identity matrix.
        Q can instead be given as C^T * C. In this case, Q needs to be
        `None`, and C not `None`.
        B * R^{-1} B^T can instead be given by G. In this case, B and R
        need to be `None`, and G not `None`.
        If R and G are `None`, then R is taken to be the identity
        matrix.
        If trans is `True`, then the dual Riccati equation is solved

        .. math::
            A X E^T + E X A^T - E X C^T R^{-1} C X E^T + Q = 0,

        where Q can be replaced by B * B^T and C^T * R^{-1} * C by G.

        This uses the `slycot` package, in particular its interfaces to
        SLICOT functions `SB02MD` (for the standard Riccati equations)
        and `SG02AD` (for the generalized Riccati equations).
        These methods are only applicable to medium-sized dense
        problems and need access to the matrix data of all operators.

        Parameters
        ----------
        A
            The |Operator| A.
        B
            The |Operator| B or `None`.
        E
            The |Operator| E or `None`.
        Q
            The |Operator| Q or `None`.
        C
            The |Operator| C or `None`.
        R
            The |Operator| R or `None`.
        G
            The |Operator| G or `None`.
        trans
            If the dual equation needs to be solved.
        options
            The |solver_options| to use (see
            :func:`ricc_solver_options`).

        Returns
        -------
        Z
            Low-rank factor of the Riccati equation solution,
            |VectorArray| from `A.source`.
        """
        _solve_ricc_check_args(A, E, B, Q, C, R, G, trans)
        options = _parse_options(options, ricc_solver_options(), 'slycot', None, False)
        assert options['type'] == 'slycot'

        A_mat = to_matrix(A, format='dense')
        B_mat = to_matrix(B, format='dense') if B else None
        C_mat = to_matrix(C, format='dense') if C else None
        R_mat = to_matrix(R, format='dense') if R else None
        G_mat = to_matrix(G, format='dense') if G else None
        Q_mat = to_matrix(Q, format='dense') if Q else None

        n = A_mat.shape[0]
        dico = 'C'

        if E is None:
            if not trans:
                if G is None:
                    if R is None:
                        G_mat = B_mat.dot(B_mat.T)
                    else:
                        G_mat = slycot.sb02mt(n, B_mat.shape[1], B_mat, R_mat)[-1]
                if C is not None:
                    Q_mat = C_mat.T.dot(C_mat)
                X = slycot.sb02md(n, A_mat, G_mat, Q_mat, dico)[0]
            else:
                if G is None:
                    if R is None:
                        G_mat = C_mat.T.dot(C_mat)
                    else:
                        G_mat = slycot.sb02mt(n, C_mat.shape[0], C_mat.T, R_mat)[-1]
                if B is not None:
                    Q_mat = B_mat.dot(B_mat.T)
                X = slycot.sb02md(n, A_mat.T, G_mat, Q_mat, dico)[0]
        else:
            E_mat = to_matrix(E, format='dense') if E else None
            jobb = 'B' if G is None else 'B'
            fact = 'C' if Q is None else 'N'
            uplo = 'U'
            jobl = 'Z'
            scal = 'N'
            sort = 'S'
            acc = 'R'
            if not trans:
                m = 0 if B is None else B_mat.shape[1]
                p = 0 if C is None else C_mat.shape[0]
                if G is not None:
                    B_mat = G_mat
                    R_mat = np.empty((1, 1))
                elif R is None:
                    R_mat = np.eye(m)
                if Q is None:
                    Q_mat = C_mat
                L_mat = np.empty((n, m))
                ret = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p,
                                    A_mat, E_mat, B_mat, Q_mat, R_mat, L_mat)
            else:
                m = 0 if C is None else C_mat.shape[0]
                p = 0 if B is None else B_mat.shape[1]
                if G is not None:
                    C_mat = G_mat
                    R_mat = np.empty((1, 1))
                elif R is None:
                    C_mat = C_mat.T
                    R_mat = np.eye(m)
                else:
                    C_mat = C_mat.T
                if Q is None:
                    Q_mat = B_mat.T
                L_mat = np.empty((n, m))
                ret = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p,
                                    A_mat.T, E_mat.T, C_mat, Q_mat, R_mat, L_mat)
            X = ret[1]
            iwarn = ret[-1]
            if iwarn == 1:
                print('slycot.sg02ad warning: solution may be inaccurate.')

        Z = chol(X)
        Z = A.source.from_numpy(np.array(Z).T)

        return Z

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.constructions import IdentityOperator
from pymor.solvers.matrix_equations.equations import BernoulliEquation


def bernoulli_stabilize(A, E, B, ast_spectrum, trans=False, solver=None):
    """Compute Bernoulli stabilizing feedback.

    Returns a matrix :math:`K` that stabilizes the spectrum of the matrix pair
    :math:`(A, E)`:

    - if trans is `True` the spectrum of

      .. math::
          (A - B K, E)

      contains the eigenvalues of :math:`(A, E)` where anti-stable eigenvalues have
      been mirrored on the imaginary axis.

    - if trans is `False` the spectrum of

      .. math::
          (A - K B, E)

      contains the eigenvalues of :math:`(A, E)` where anti-stable eigenvalues have
      been mirrored on the imaginary axis.

    See e.g. :cite:`BBQ07`.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E.
    B
        The operator B as a |VectorArray|.
    ast_spectrum
        Tuple `(lev, ew, rev)` where `ew` contains the anti-stable eigenvalues
        and `lev` and `rev` are |VectorArrays| representing the eigenvectors.
    trans
        Indicates which stabilization to perform.
    solver
        The |BernoulliSolver| for the projected |BernoulliEquation| or `None`,
        then :class:`~pymor.solvers.matrix_equations.default.DefaultBernoulliSolver`
        is used.

    Returns
    -------
    K
        The stabilizing feedback as a |VectorArray|.
    """
    if E is None:
        E = IdentityOperator(A.source)

    ast_levs = ast_spectrum[0]
    ast_revs = ast_spectrum[2]

    Mt = E.apply2(ast_levs, ast_revs)
    At = A.apply2(ast_levs, ast_revs)

    if trans:
        Bt = ast_levs.inner(B)
    else:
        Bt = B.inner(ast_revs)

    Xz = BernoulliEquation.from_matrices(At, Mt, Bt, trans=trans).solve(solver=solver).conj()

    if trans:
        K = E.apply_adjoint(ast_levs.conj()).lincomb(Xz @ B.inner(ast_levs).T)
    else:
        K = E.apply_adjoint(ast_revs.conj()).lincomb(Xz @ B.inner(ast_revs).T)

    return K.real

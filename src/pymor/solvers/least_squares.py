# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.solvers.interface import Solver


class QRLeastSquaresSolver(Solver):
    """Least-squares solver using QR decomposition.

    Convertes `operator` to a |VectorArray| via
    :meth:`~pymor.operators.interface.Operator.as_range_array` or
    :meth:`~pymor.operators.interface.Operator.as_source_array`.
    Then uses :func:`~pymor.algorithms.gram_schmidt.gram_schmidt`
    for QR decomposition.

    Parameters
    ----------
    space
        If `'range'`, represent `operator` as a |VectorArray| in
        `operator.range`. If `'source'`, represent `operator` as a
        |VectorArray| in `operator.source`.
    """

    least_squares = True

    def __init__(self, space='range'):
        assert space in {'range', 'source'}
        self.__auto_init(locals())

    def _solve(self, operator, V, mu, initial_guess):
        if self.space == 'range':
            array = operator.as_range_array(mu)
            Q, R = gram_schmidt(array, return_R=True, reiterate=False)
            v = Q.inner(V)
            u = spla.lstsq(R, v)[0]
            U = operator.source.make_array(u)
        elif self.space == 'source':
            array = operator.as_source_array(mu)
            Q, R = gram_schmidt(array, return_R=True, reiterate=False)
            v = spla.lstsq(R.T.conj(), V.to_numpy())[0]
            U = Q.lincomb(v)
        else:
            assert False

        return U, {}

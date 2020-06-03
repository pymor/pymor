# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.list import ListVectorArrayOperatorBase
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.list import NumpyListVectorSpace


class NumpyListVectorArrayMatrixOperator(ListVectorArrayOperatorBase, NumpyMatrixOperator):
    """Variant of |NumpyMatrixOperator| using |ListVectorArray| instead of |NumpyVectorArray|.

    This class is mainly intended for performance tests of |ListVectorArray|.
    In general |NumpyMatrixOperator| should be used instead of this class.

    Parameters
    ----------
    matrix
        The |NumPy array| which is to be wrapped.
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    def __init__(self, matrix, source_id=None, range_id=None, solver_options=None, name=None):
        super().__init__(matrix, source_id=source_id, range_id=range_id, solver_options=solver_options, name=name)
        self.source = NumpyListVectorSpace(matrix.shape[1], source_id)
        self.range = NumpyListVectorSpace(matrix.shape[0], range_id)

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        return self.matrix.dot(u._array)

    def _apply_inverse_one_vector(self, v, mu=None, least_squares=False, prepare_data=None):
        op = self.with_(new_type=NumpyMatrixOperator)
        u = op.apply_inverse(
            op.range.make_array(v._array),
            least_squares=least_squares
        ).to_numpy().ravel()

        return u

    def apply_adjoint(self, V, mu=None):
        return NumpyMatrixOperator.apply_adjoint(self, V, mu=mu)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        return NumpyMatrixOperator.apply_inverse_adjoint(self, U, mu=mu, least_squares=least_squares)

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
        lincomb = super()._assemble_lincomb(operators, coefficients, identity_shift)
        if lincomb is None:
            return None
        else:
            return lincomb.with_(new_type=NumpyListVectorArrayMatrixOperator)

# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import abstractmethod
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.list import NumpyListVectorSpace


class ListVectorArrayOperatorBase(Operator):
    """Base |Operator| for |ListVectorArrays|."""

    @abstractmethod
    def _apply_one_vector(self, u, mu=None):
        pass

    def _apply_adjoint_one_vector(self, v, mu=None):
        raise NotImplementedError

    def apply(self, U, mu=None):
        assert U in self.source
        V = [self._apply_one_vector(u, mu=mu) for u in U.vectors]
        return self.range.make_array(V)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        try:
            U = [self._apply_adjoint_one_vector(v, mu=mu) for v in V.vectors]
        except NotImplementedError:
            return super().apply_adjoint(V, mu=mu)
        return self.source.make_array(U)


class LinearComplexifiedListVectorArrayOperatorBase(ListVectorArrayOperatorBase):
    """Base |Operator| for complexified |ListVectorArrays|."""

    linear = True

    @abstractmethod
    def _real_apply_one_vector(self, u, mu=None):
        pass

    def _real_apply_adjoint_one_vector(self, v, mu=None):
        raise NotImplementedError

    def _apply_one_vector(self, u, mu=None):
        real_part = self._real_apply_one_vector(u.real_part, mu=mu)
        if u.imag_part is not None:
            imag_part = self._real_apply_one_vector(u.imag_part, mu=mu)
        else:
            imag_part = None
        return self.range.vector_type(real_part, imag_part)

    def _apply_adjoint_one_vector(self, v, mu=None):
        real_part = self._real_apply_adjoint_one_vector(v.real_part, mu=mu)
        if v.imag_part is not None:
            imag_part = self._real_apply_adjoint_one_vector(v.imag_part, mu=mu)
        else:
            imag_part = None
        return self.source.vector_type(real_part, imag_part)


class NumpyListVectorArrayMatrixOperator(ListVectorArrayOperatorBase, NumpyMatrixOperator):
    """Variant of |NumpyMatrixOperator| using |ListVectorArray| instead of |NumpyVectorArray|.

    This class is mainly intended for performance tests of |ListVectorArray|.
    In general |NumpyMatrixOperator| should be used instead of this class.

    Parameters
    ----------
    matrix
        The |NumPy array| which is to be wrapped.
    solver
        The |Solver| for the operator.
    name
        Name of the operator.
    """

    def __init__(self, matrix, solver=None, name=None):
        super().__init__(matrix, solver=solver, name=name)
        self.source = NumpyListVectorSpace(matrix.shape[1])
        self.range = NumpyListVectorSpace(matrix.shape[0])

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        return self.matrix.dot(u._array)

    # TODO: update
    def _apply_inverse_one_vector(self, v, mu, initial_guess, prepare_data):
        op = self.with_(new_type=NumpyMatrixOperator)
        u = op.apply_inverse(
            op.range.make_array(v._array),
            initial_guess=op.source.make_array(initial_guess._array) if initial_guess is not None else None
        ).to_numpy().ravel()

        return u

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _apply_inverse_adjoint_one_vector(self, u, mu, initial_guess, prepare_data):
        raise NotImplementedError

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., name=None):
        lincomb = super()._assemble_lincomb(operators, coefficients, identity_shift=identity_shift, name=name)
        if lincomb is None:
            return None
        else:
            return lincomb.with_(new_type=NumpyListVectorArrayMatrixOperator)

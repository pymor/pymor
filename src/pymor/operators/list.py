# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import abstractmethod
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.list import NumpyListVectorSpace


class ListVectorArrayOperatorBase(Operator):

    def _prepare_apply(self, U, mu, kind, least_squares=False):
        pass

    @abstractmethod
    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        pass

    def _apply_inverse_one_vector(self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def apply(self, U, mu=None):
        assert U in self.source
        data = self._prepare_apply(U, mu, 'apply')
        V = [self._apply_one_vector(u, mu=mu, prepare_data=data) for u in U.vectors]
        return self.range.make_array(V)

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        assert V in self.range
        assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)
        try:
            data = self._prepare_apply(V, mu, 'apply_inverse', least_squares=least_squares)
            U = [self._apply_inverse_one_vector(v, mu=mu,
                                                initial_guess=(initial_guess.vectors[i]
                                                               if initial_guess is not None else None),
                                                least_squares=least_squares, prepare_data=data)
                 for i, v in enumerate(V.vectors)]
        except NotImplementedError:
            return super().apply_inverse(V, mu=mu, least_squares=least_squares)
        return self.source.make_array(U)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        try:
            data = self._prepare_apply(V, mu, 'apply_adjoint')
            U = [self._apply_adjoint_one_vector(v, mu=mu, prepare_data=data) for v in V.vectors]
        except NotImplementedError:
            return super().apply_adjoint(V, mu=mu)
        return self.source.make_array(U)

    def apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        assert U in self.source
        try:
            data = self._prepare_apply(U, mu, 'apply_inverse_adjoint', least_squares=least_squares)
            V = [self._apply_inverse_adjoint_one_vector(u, mu=mu,
                                                        initial_guess=(initial_guess.vectors[i]
                                                                       if initial_guess is not None else None),
                                                        least_squares=least_squares, prepare_data=data)
                 for i, u in enumerate(U.vectors)]
        except NotImplementedError:
            return super().apply_inverse_adjoint(U, mu=mu, least_squares=least_squares)
        return self.range.make_array(V)


class LinearComplexifiedListVectorArrayOperatorBase(ListVectorArrayOperatorBase):

    linear = True

    @abstractmethod
    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        pass

    def _real_apply_inverse_one_vector(self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _real_apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None, least_squares=False,
                                               prepare_data=None):
        raise NotImplementedError

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        real_part = self._real_apply_one_vector(u.real_part, mu=mu, prepare_data=prepare_data)
        if u.imag_part is not None:
            imag_part = self._real_apply_one_vector(u.imag_part, mu=mu, prepare_data=prepare_data)
        else:
            imag_part = None
        return self.range.vector_type(real_part, imag_part)

    def _apply_inverse_one_vector(self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        real_part = self._real_apply_inverse_one_vector(v.real_part, mu=mu,
                                                        initial_guess=(initial_guess.real_part
                                                                       if initial_guess is not None else None),
                                                        least_squares=least_squares,
                                                        prepare_data=prepare_data)
        if v.imag_part is not None:
            imag_part = self._real_apply_inverse_one_vector(v.imag_part, mu=mu,
                                                            initial_guess=(initial_guess.imag_part
                                                                           if initial_guess is not None else None),
                                                            least_squares=least_squares,
                                                            prepare_data=prepare_data)
        else:
            imag_part = None
        return self.source.vector_type(real_part, imag_part)

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        real_part = self._real_apply_adjoint_one_vector(v.real_part, mu=mu, prepare_data=prepare_data)
        if v.imag_part is not None:
            imag_part = self._real_apply_adjoint_one_vector(v.imag_part, mu=mu, prepare_data=prepare_data)
        else:
            imag_part = None
        return self.source.vector_type(real_part, imag_part)

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        real_part = self._real_apply_inverse_adjoint_one_vector(u.real_part, mu=mu,
                                                                initial_guess=(initial_guess.real_part
                                                                               if initial_guess is not None else None),
                                                                least_squares=least_squares,
                                                                prepare_data=prepare_data)
        if u.imag_part is not None:
            imag_part = self._real_apply_inverse_adjoint_one_vector(u.imag_part, mu=mu,
                                                                    initial_guess=(initial_guess.imag_part
                                                                                   if initial_guess is not None
                                                                                   else None),
                                                                    least_squares=least_squares,
                                                                    prepare_data=prepare_data)
        else:
            imag_part = None
        return self.range.vector_type(real_part, imag_part)


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

    def _apply_inverse_one_vector(self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        op = self.with_(new_type=NumpyMatrixOperator)
        u = op.apply_inverse(
            op.range.make_array(v._array),
            initial_guess=op.source.make_array(initial_guess._array) if initial_guess is not None else None,
            least_squares=least_squares
        ).to_numpy().ravel()

        return u

    def apply_adjoint(self, V, mu=None):
        return NumpyMatrixOperator.apply_adjoint(self, V, mu=mu)

    def apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        return NumpyMatrixOperator.apply_inverse_adjoint(self, U, mu=mu,
                                                         initial_guess=initial_guess, least_squares=least_squares)

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
        lincomb = super()._assemble_lincomb(operators, coefficients, identity_shift)
        if lincomb is None:
            return None
        else:
            return lincomb.with_(new_type=NumpyListVectorArrayMatrixOperator)

# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import abstractmethod
from pymor.operators.interface import Operator


class ListVectorArrayOperatorBase(Operator):

    def _prepare_apply(self, U, mu, kind, least_squares=False):
        pass

    @abstractmethod
    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        pass

    def _apply_inverse_one_vector(self, v, mu=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def apply(self, U, mu=None):
        assert U in self.source
        data = self._prepare_apply(U, mu, 'apply')
        V = [self._apply_one_vector(u, mu=mu, prepare_data=data) for u in U._list]
        return self.range.make_array(V)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        try:
            data = self._prepare_apply(V, mu, 'apply_inverse', least_squares=least_squares)
            U = [self._apply_inverse_one_vector(v, mu=mu, least_squares=least_squares, prepare_data=data)
                 for v in V._list]
        except NotImplementedError:
            return super().apply_inverse(V, mu=mu, least_squares=least_squares)
        return self.source.make_array(U)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        try:
            data = self._prepare_apply(V, mu, 'apply_adjoint')
            U = [self._apply_adjoint_one_vector(v, mu=mu, prepare_data=data) for v in V._list]
        except NotImplementedError:
            return super().apply_adjoint(V, mu=mu)
        return self.source.make_array(U)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        assert U in self.source
        try:
            data = self._prepare_apply(U, mu, 'apply_inverse_adjoint', least_squares=least_squares)
            V = [self._apply_inverse_adjoint_one_vector(u, mu=mu, least_squares=least_squares, prepare_data=data)
                 for u in U._list]
        except NotImplementedError:
            return super().apply_inverse_adjoint(U, mu=mu, least_squares=least_squares)
        return self.range.make_array(V)


class LinearComplexifiedListVectorArrayOperatorBase(ListVectorArrayOperatorBase):

    linear = True

    @abstractmethod
    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        pass

    def _real_apply_inverse_one_vector(self, v, mu=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _real_apply_inverse_adjoint_one_vector(self, u, mu=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        real_part = self._real_apply_one_vector(u.real_part, mu=mu, prepare_data=prepare_data)
        if u.imag_part is not None:
            imag_part = self._real_apply_one_vector(u.imag_part, mu=mu, prepare_data=prepare_data)
        else:
            imag_part = None
        return self.range.complexified_vector_type(real_part, imag_part)

    def _apply_inverse_one_vector(self, v, mu=None, least_squares=False, prepare_data=None):
        real_part = self._real_apply_inverse_one_vector(v.real_part, mu=mu, least_squares=least_squares,
                                                        prepare_data=prepare_data)
        if v.imag_part is not None:
            imag_part = self._real_apply_inverse_one_vector(v.imag_part, mu=mu, least_squares=least_squares,
                                                            prepare_data=prepare_data)
        else:
            imag_part = None
        return self.source.complexified_vector_type(real_part, imag_part)

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        real_part = self._real_apply_adjoint_one_vector(v.real_part, mu=mu, prepare_data=prepare_data)
        if v.imag_part is not None:
            imag_part = self._real_apply_adjoint_one_vector(v.imag_part, mu=mu, prepare_data=prepare_data)
        else:
            imag_part = None
        return self.source.complexified_vector_type(real_part, imag_part)

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, least_squares=False, prepare_data=None):
        real_part = self._real_apply_inverse_adjoint_one_vector(u.real_part, mu=mu, least_squares=least_squares,
                                                                prepare_data=prepare_data)
        if u.imag_part is not None:
            imag_part = self._real_apply_inverse_adjoint_one_vector(u.imag_part, mu=mu, least_squares=least_squares,
                                                                    prepare_data=prepare_data)
        else:
            imag_part = None
        return self.range.complexified_vector_type(real_part, imag_part)

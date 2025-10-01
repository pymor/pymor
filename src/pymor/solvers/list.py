# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.solvers.interface import Solver


class ListVectorArrayBasedSolver(Solver):

    def _prepare(self, operator, U, mu, adjoint):
        pass

    def _solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        raise NotImplementedError

    def _solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        raise NotImplementedError

    def _solve(self, operator, V, mu, initial_guess):
        # TODO: support return_info
        try:
            data = self._prepare(operator, V, mu, False)
            U = [self._solve_one_vector(operator, v, mu=mu,
                                        initial_guess=(initial_guess.vectors[i]
                                                       if initial_guess is not None else None),
                                        prepare_data=data)
                 for i, v in enumerate(V.vectors)]
        except NotImplementedError as e:
            raise NotImplementedError from e
        return operator.source.make_array(U), {}

    def _solve_adjoint(self, operator, U, mu, initial_guess):
        try:
            data = self._prepare(operator, U, mu, True)
            V = [self._solve_adjoint_one_vector(operator, u, mu=mu,
                                                initial_guess=(initial_guess.vectors[i]
                                                               if initial_guess is not None else None),
                                                prepare_data=data)
                 for i, u in enumerate(U.vectors)]
        except NotImplementedError as e:
            raise NotImplementedError from e
        return operator.range.make_array(V), {}


class ComplexifiedListVectorArrayBasedSolver(ListVectorArrayBasedSolver):
    def _real_solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        raise NotImplementedError

    def _real_solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        raise NotImplementedError

    def _solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        real_part = self._real_solve_one_vector(operator, v.real_part, mu=mu,
                                                initial_guess=(initial_guess.real_part
                                                               if initial_guess is not None else None),
                                                prepare_data=prepare_data)
        if v.imag_part is not None:
            imag_part = self._real_solve_one_vector(operator, v.imag_part, mu=mu,
                                                    initial_guess=(initial_guess.imag_part
                                                                   if initial_guess is not None else None),
                                                    prepare_data=prepare_data)
        else:
            imag_part = None
        return operator.source.vector_type(real_part, imag_part)

    def _solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        real_part = self._real_solve_adjoint_one_vector(operator, u.real_part, mu=mu,
                                                        initial_guess=(initial_guess.real_part
                                                                       if initial_guess is not None else None),
                                                        prepare_data=prepare_data)
        if u.imag_part is not None:
            imag_part = self._real_solve_adjoint_one_vector(operator, u.imag_part, mu=mu,
                                                            initial_guess=(initial_guess.imag_part
                                                                           if initial_guess is not None
                                                                           else None),
                                                            prepare_data=prepare_data)
        else:
            imag_part = None
        return operator.range.vector_type(real_part, imag_part)

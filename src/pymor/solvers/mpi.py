# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.solvers.interface import Solver
from pymor.tools import mpi
from pymor.vectorarrays.mpi import _indexed


class MPISolver(Solver):
    """MPI distributed |Solver|.

    Forwards calls to MPI-aware solvers on each MPI rank.

    Parameters
    ----------
    obj_id
        :class:`~pymor.tools.mpi.ObjectId` of the |Solver|
        to wrap. If `None`, the `operator`'s default solver is
        used.
    """

    def __init__(self, obj_id=None):
        self.__auto_init(locals())

    def _solve(self, operator, V, mu, initial_guess):
        op = operator.assemble(mu)
        return operator.source.make_array(mpi.call(mpi.function_call_manage, _MPISolver_solve,
                                                   self.obj_id, op.obj_id, V.impl.obj_id, V.ind, mu,
                                                   initial_guess=(initial_guess.impl.obj_id if initial_guess is not None
                                                                  else None))), {}

    def _solve_adjoint(self, operator, U, mu, initial_guess):
        operator = operator.assemble(mu)
        return operator.source.make_array(mpi.call(mpi.function_call_manage, _MPISolver_solve_adjoint,
                                                   self.obj_id, operator.obj_id, U.impl.obj_id, U.ind, mu,
                                                   initial_guess=(initial_guess.impl.obj_id if initial_guess is not None
                                                                  else None))), {}


def _MPISolver_solve(solver, operator, V, V_ind, mu, initial_guess):
    return operator.apply_inverse(_indexed(V, V_ind), mu=mu, initial_guess=initial_guess, solver=solver)


def _MPISolver_solve_adjoint(solver, operator, U, U_ind, mu, initial_guess):
    return operator.apply_inverse_adjoint(_indexed(U, U_ind), mu=mu, initial_guess=initial_guess, solver=solver)

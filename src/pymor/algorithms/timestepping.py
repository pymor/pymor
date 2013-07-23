# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core import BasicInterface, abstractmethod
from pymor.la import VectorArrayInterface
from pymor.operators import OperatorInterface, MatrixBasedOperatorInterface


class TimeStepperInterface(BasicInterface):

    @abstractmethod
    def solve(initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None):
        pass


class ExplicitEulerTimeStepper(TimeStepperInterface):

    def __init__(self, nt):
        self.nt = nt
        self.lock()

    with_arguments = set(('nt',))
    def with_(self, **kwargs):
        return self._with_via_init(kwargs)

    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None):
        if mass is not None:
            raise NotImplementedError
        if rhs is None:
            raise NotImplementedError
        return explicit_euler(operator, rhs, initial_data, initial_time, end_time, self.nt, mu)


def implicit_euler(A, F, M, U0, t0, t1, nt, mu=None, invert_options=None):
    assert isinstance(A, OperatorInterface)
    assert isinstance(F, (MatrixBasedOperatorInterface, VectorArrayInterface))
    assert isinstance(M, OperatorInterface)
    assert not M.parametric
    assert A.dim_source == A.dim_range
    assert M.dim_source == M.dim_range == A.dim_source

    dt = (t1 - t0) / nt

    if isinstance(F, MatrixBasedOperatorInterface):
        assert F.dim_range == 1
        assert F.dim_source == A.dim_source
        F_time_dep = F.parametric and '_t' in F.parameter_type
        if not F_time_dep:
            dt_F = F.as_vector_array(mu) * dt
    else:
        assert len(F) == 1
        assert F.dim == A.dim_source
        F_time_dep = False
        dt_F = F * dt

    assert isinstance(U0, VectorArrayInterface)
    assert len(U0) == 1
    assert U0.dim == A.dim_source

    A_time_dep = A.parametric and '_t' in A.parameter_type

    R = A.type_source.empty(A.dim_source, reserve=nt+1)
    R.append(U0)

    M_dt_A = M + A * dt
    if isinstance(M_dt_A, MatrixBasedOperatorInterface) and not A_time_dep:
        M_dt_A = M_dt_A.assemble(mu)

    t = t0
    U = U0.copy()

    for n in xrange(nt):
        t += dt
        mu['_t'] = t
        if F_time_dep:
            dt_F = F.as_vector_array(mu) * dt
        U = M_dt_A.apply_inverse(M.apply(U) + dt_F, mu=mu, options=invert_options)
        R.append(U)

    return R


def explicit_euler(A, F, U0, t0, t1, nt, mu=None):
    assert isinstance(A, OperatorInterface)
    assert isinstance(F, (MatrixBasedOperatorInterface, VectorArrayInterface))
    assert A.dim_source == A.dim_range

    if isinstance(F, MatrixBasedOperatorInterface):
        assert F.dim_range == 1
        assert F.dim_source == A.dim_source
        F_time_dep = F.parametric and '_t' in F.parameter_type
        if not F_time_dep:
            F_ass = F.as_vector_array(mu)
    else:
        assert len(F) == 1
        assert F.dim == A.dim_source
        F_time_dep = False
        F_ass = F

    assert isinstance(U0, VectorArrayInterface)
    assert len(U0) == 1
    assert U0.dim == A.dim_source

    A_time_dep = A.parametric and '_t' in A.parameter_type
    if isinstance(A, MatrixBasedOperatorInterface) and not A_time_dep:
        A = A.assemble(mu)

    dt = (t1 - t0) / nt
    R = A.type_source.empty(A.dim_source, reserve=nt+1)
    R.append(U0)

    t = t0
    U = U0.copy()

    for n in xrange(nt):
        t += dt
        mu['_t'] = t
        if F_time_dep:
            F_ass = F.assemble(mu).as_vector_array()
        U += (F_ass - A.apply(U, mu=mu)) * dt
        R.append(U)

    return R

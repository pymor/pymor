# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core import ImmutableInterface, abstractmethod
from pymor.la import VectorArrayInterface
from pymor.operators import OperatorInterface


class TimeStepperInterface(ImmutableInterface):

    @abstractmethod
    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None):
        pass


class ImplicitEulerTimeStepper(TimeStepperInterface):

    def __init__(self, nt, invert_options=None):
        self.nt = nt
        self.invert_options = invert_options

    with_arguments = set(('nt',))
    def with_(self, **kwargs):
        return self._with_via_init(kwargs)

    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None):
        return implicit_euler(operator, rhs, mass, initial_data, initial_time, end_time, self.nt, mu,
                              self.invert_options)


class ExplicitEulerTimeStepper(TimeStepperInterface):

    def __init__(self, nt):
        self.nt = nt

    with_arguments = set(('nt',))
    def with_(self, **kwargs):
        return self._with_via_init(kwargs)

    def solve(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None):
        if mass is not None:
            raise NotImplementedError
        return explicit_euler(operator, rhs, initial_data, initial_time, end_time, self.nt, mu)


def implicit_euler(A, F, M, U0, t0, t1, nt, mu=None, invert_options=None):
    assert isinstance(A, OperatorInterface)
    assert isinstance(F, (OperatorInterface, VectorArrayInterface))
    assert isinstance(M, OperatorInterface)
    assert not M.parametric
    assert A.dim_source == A.dim_range
    assert M.dim_source == M.dim_range == A.dim_source

    dt = (t1 - t0) / nt

    if isinstance(F, OperatorInterface):
        assert F.dim_range == 1
        assert F.dim_source == A.dim_source
        F_time_dep = F.parametric and '_t' in F.parameter_type
        if not F_time_dep:
            dt_F = F.as_vector(mu) * dt
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
    if hasattr(M_dt_A, 'assemble') and not A_time_dep:
        M_dt_A = M_dt_A.assemble(mu)

    t = t0
    U = U0.copy()

    for n in xrange(nt):
        t += dt
        mu['_t'] = t
        if F_time_dep:
            dt_F = F.as_vector(mu) * dt
        U = M_dt_A.apply_inverse(M.apply(U) + dt_F, mu=mu, options=invert_options)
        R.append(U)

    return R


def explicit_euler(A, F, U0, t0, t1, nt, mu=None):
    assert isinstance(A, OperatorInterface)
    assert F is None or isinstance(F, (OperatorInterface, VectorArrayInterface))
    assert A.dim_source == A.dim_range

    if isinstance(F, OperatorInterface):
        assert F.dim_range == 1
        assert F.dim_source == A.dim_source
        F_time_dep = F.parametric and '_t' in F.parameter_type
        if not F_time_dep:
            F_ass = F.as_vector(mu)
    elif isinstance(F, VectorArrayInterface):
        assert len(F) == 1
        assert F.dim == A.dim_source
        F_time_dep = False
        F_ass = F

    assert isinstance(U0, VectorArrayInterface)
    assert len(U0) == 1
    assert U0.dim == A.dim_source

    A_time_dep = A.parametric and '_t' in A.parameter_type
    if hasattr(A, 'assemble') and not A_time_dep:
        A = A.assemble(mu)

    dt = (t1 - t0) / nt
    R = A.type_source.empty(A.dim_source, reserve=nt+1)
    R.append(U0)

    t = t0
    U = U0.copy()

    if F is None:
        for n in xrange(nt):
            t += dt
            mu['_t'] = t
            U.axpy(-dt, A.apply(U, mu=mu))
            R.append(U)
    else:
        for n in xrange(nt):
            t += dt
            mu['_t'] = t
            if F_time_dep:
                F_ass = F.as_vector(mu)
            U.axpy(dt, F_ass - A.apply(U, mu=mu))
            R.append(U)

    return R

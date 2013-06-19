# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.la import VectorArrayInterface
from pymor.operators import OperatorInterface, LinearOperatorInterface
from pymor.operators.solvers import solve_linear


def implicit_euler(A, F, M, U0, t0, t1, nt, mu_A=None, mu_F=None, solver=None):
    assert isinstance(A, LinearOperatorInterface)
    assert A.dim_source == A.dim_range
    A_time_dep = A.parameter_type and '_t' in A.parameter_type
    if not A_time_dep:
        A = A.assemble(mu_A)

    assert isinstance(F, (LinearOperatorInterface, VectorArrayInterface))
    if isinstance(F, LinearOperatorInterface):
        assert F.dim_range == 1
        assert F.dim_source == A.dim_source
        F_time_dep = F.parameter_type and '_t' in F.parameter_type
        if not F_time_dep:
            F = F.assemble(mu_F).as_vector_array()
    else:
        assert len(F) == 1
        assert F.dim == A.dim_source
        F_time_dep = False

    if M:
        assert isinstance(M, LinearOperatorInterface)
        assert M.dim_source == M.dim_range == A.dim_source
        M = M.assemble()

    assert isinstance(U0, VectorArrayInterface)
    assert len(U0) == 1
    assert U0.dim == A.dim_source

    solver = solver or solve_linear

    dt = (t1 - t0) / nt
    R = A.type_source.empty(A.dim_source, reserve=nt+1)
    R.append(U0)

    if not A_time_dep:
        M_dt_A = M + A * dt
    if not F_time_dep:
        dt_F = F * dt

    t = t0
    U = U0.copy()

    for n in xrange(nt):
        t += dt
        if A_time_dep:
            mu_A['_t'] = t
            M_dt_A = M + A.assemble(mu_A) * dt
        if F_time_dep:
            mu_F['_t'] = t
            dt_F = F.assemble(mu_F).as_vector_array() * dt
        U = solver(M_dt_A, M.apply(U) + dt_F)
        R.append(U)

    return R


def explicit_euler(A, F, U0, t0, t1, nt, mu_A=None, mu_F=None):
    assert isinstance(A, OperatorInterface)
    assert A.dim_source == A.dim_range
    A_time_dep = A.parameter_type and '_t' in A.parameter_type

    assert isinstance(F, (LinearOperatorInterface, VectorArrayInterface))
    if isinstance(F, LinearOperatorInterface):
        assert F.dim_range == 1
        assert F.dim_source == A.dim_source
        F_time_dep = F.parameter_type and '_t' in F.parameter_type
        if not F_time_dep:
            F_ass = F.assemble(mu_F).as_vector_array()
    else:
        assert len(F) == 1
        assert F.dim == A.dim_source
        F_ass = F
        F_time_dep = False

    assert isinstance(U0, VectorArrayInterface)
    assert len(U0) == 1
    assert U0.dim == A.dim_source

    dt = (t1 - t0) / nt
    R = A.type_source.empty(A.dim_source, reserve=nt+1)
    R.append(U0)

    t = t0
    U = U0.copy()

    for n in xrange(nt):
        t += dt
        if A_time_dep:
            mu_A['_t'] = t
        if F_time_dep:
            mu_F['_t'] = t
            F_ass = F.assemble(mu_F).as_vector_array()
        U = U + (F_ass - A.apply(U, mu=mu_A)) * dt
        R.append(U)

    return R

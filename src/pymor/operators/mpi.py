# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.operators.interface import Operator
from pymor.tools import mpi
from pymor.vectorarrays.mpi import MPIVectorSpace, _register_local_space, _indexed


class MPIOperator(Operator):
    """MPI distributed |Operator|.

    Given a single-rank implementation of an |Operator|, this
    wrapper class uses the event loop from :mod:`pymor.tools.mpi`
    to allow an MPI distributed usage of the |Operator|.

    Instances of `MPIOperator` can be used on rank 0 like any
    other (non-distributed) |Operator|.

    Note, however, that the underlying |Operator| implementation
    needs to be MPI aware. For instance, the operator's `apply`
    method has to perform the necessary MPI communication to
    obtain all DOFs hosted on other MPI ranks which are required
    for the local operator evaluation.

    Instead of instantiating :class:`MPIOperator` directly, it
    is usually preferable to use :func:`mpi_wrap_operator` instead.

    Parameters
    ----------
    obj_id
        :class:`~pymor.tools.mpi.ObjectId` of the local |Operators|
        on each rank.
    mpi_range
        Set to `True` if the range of the |Operator| is MPI distributed.
    mpi_source
        Set to `True` if the source of the |Operator| is MPI distributed.
    with_apply2
        Set to `True` if the operator implementation has its own
        MPI aware implementation of `apply2` and `pairwise_apply2`.
        Otherwise, the default implementations using `apply` and
        :meth:`~pymor.vectorarrays.interface.VectorArray.inner`
        will be used.
    pickle_local_spaces
        If `pickle_local_spaces` is `False`, a unique identifier
        is computed for each local source/range |VectorSpace|, which is then
        transferred to rank 0 instead of the true |VectorSpace|. This
        allows the usage of :class:`~pymor.vectorarrays.mpi.MPIVectorArray`
        even when the local |VectorSpaces| are not picklable.
    space_type
        This class will be used to wrap the local |VectorArrays|
        returned by the local operators into an MPI distributed
        |VectorArray| managed from rank 0. By default,
        :class:`~pymor.vectorarrays.mpi.MPIVectorSpace` will be used,
        other options are :class:`~pymor.vectorarrays.mpi.MPIVectorSpaceAutoComm`
        and :class:`~pymor.vectorarrays.mpi.MPIVectorSpaceNoComm`.
    """

    def __init__(self, obj_id, mpi_range, mpi_source, with_apply2=False, pickle_local_spaces=True,
                 space_type=MPIVectorSpace):
        assert mpi_source or mpi_range

        self.__auto_init(locals())
        self.op = op = mpi.get_object(obj_id)
        self.linear = op.linear
        self.parameters = op.parameters
        self.parameters_own = op.parameters_own
        self.parameters_internal = op.parameters_internal
        self.name = op.name
        if mpi_source:
            local_spaces = mpi.call(_MPIOperator_get_local_spaces, obj_id, True, pickle_local_spaces)
            if all(ls == local_spaces[0] for ls in local_spaces):
                local_spaces = (local_spaces[0],)
            self.source = space_type(local_spaces)
        else:
            self.source = op.source
        if mpi_range:
            local_spaces = mpi.call(_MPIOperator_get_local_spaces, obj_id, False, pickle_local_spaces)
            if all(ls == local_spaces[0] for ls in local_spaces):
                local_spaces = (local_spaces[0],)
            self.range = space_type(local_spaces)
        else:
            self.range = op.range
        self.solver_options = op.solver_options

    def apply(self, U, mu=None):
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        U_ind = U.ind
        U = U.impl.obj_id if self.mpi_source else U
        if self.mpi_range:
            return self.range.make_array(
                mpi.call(mpi.function_call_manage, _MPIOperator_apply, self.obj_id, U, U_ind, mu))
        else:
            return mpi.call(mpi.function_call, _MPIOperator_apply, self.obj_id, U, U_ind, mu)

    def as_range_array(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        assert self.mpi_range
        return self.range.make_array(mpi.call(mpi.method_call_manage, self.obj_id, 'as_range_array', mu=mu))

    def as_source_array(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        assert self.mpi_source
        return self.source.make_array(mpi.call(mpi.method_call_manage, self.obj_id, 'as_source_array', mu=mu))

    def apply2(self, V, U, mu=None):
        if not self.with_apply2:
            return super().apply2(V, U, mu=mu)
        assert V in self.range
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        U_ind, V_ind = U.ind, V.ind
        U = U.impl.obj_id if self.mpi_source else U
        V = V.impl.obj_id if self.mpi_range else V
        return mpi.call(_MPIOperator_apply2, self.obj_id, V, V_ind, U, U_ind, mu)

    def pairwise_apply2(self, V, U, mu=None):
        if not self.with_apply2:
            return super().pairwise_apply2(V, U, mu=mu)
        assert V in self.range
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        U_ind, V_ind = U.ind, V.ind
        U = U.impl.obj_id if self.mpi_source else U
        V = V.impl.obj_id if self.mpi_range else V
        return mpi.call(_MPIOperator_pairwise_apply2, self.obj_id, V, V_ind, U, U_ind, mu)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        assert self.parameters.assert_compatible(mu)
        V_ind = V.ind
        V = V.impl.obj_id if self.mpi_range else V
        if self.mpi_source:
            return self.source.make_array(
                mpi.call(mpi.function_call_manage, _MPIOperator_apply_adjoint, self.obj_id, V, V_ind, mu)
            )
        else:
            return mpi.call(_MPIOperator_apply_adjoint, self.obj_id, V, V_ind, mu)

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        if not self.mpi_source or not self.mpi_range:
            raise NotImplementedError
        assert V in self.range
        assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)
        assert self.parameters.assert_compatible(mu)
        return self.source.make_array(mpi.call(mpi.function_call_manage, _MPIOperator_apply_inverse,
                                               self.obj_id,
                                               V.impl.obj_id, V.ind, mu,
                                               initial_guess=(initial_guess.impl.obj_id if initial_guess is not None
                                                              else None),
                                               least_squares=least_squares))

    def apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        if not self.mpi_source or not self.mpi_range:
            raise NotImplementedError
        assert U in self.source
        assert initial_guess is None or initial_guess in self.range and len(initial_guess) == len(U)
        assert self.parameters.assert_compatible(mu)
        return self.source.make_array(mpi.call(mpi.function_call_manage, _MPIOperator_apply_inverse_adjoint,
                                               self.obj_id,
                                               U.impl.obj_id, U.ind, mu,
                                               initial_guess=(initial_guess.impl.obj_id if initial_guess is not None
                                                              else None),
                                               least_squares=least_squares))

    def jacobian(self, U, mu=None):
        assert U in self.source
        assert self.mpi_source
        assert self.parameters.assert_compatible(mu)
        return self.with_(obj_id=mpi.call(mpi.function_call_manage, _MPIOperator_jacobian,
                                          self.obj_id, U.impl.obj_id, U.ind, mu))

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        assembled_op = mpi.call(_MPIOperator_assemble, self.obj_id, mu)
        if assembled_op is not None:
            return self.with_(obj_id=assembled_op)
        else:
            return self

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
        if not all(isinstance(op, MPIOperator) for op in operators):
            return None
        assert solver_options is None
        operators = [op.obj_id for op in operators]
        obj_id = mpi.call(_MPIOperator_assemble_lincomb, operators, coefficients, identity_shift, name=name)
        op = mpi.get_object(obj_id)
        if op is None:
            mpi.call(mpi.remove_object, obj_id)
            return None
        else:
            return self.with_(obj_id=obj_id)

    def restricted(self, dofs):
        retval = mpi.call(mpi.function_call, _MPIOperator_restriced, self.obj_id, dofs)
        if retval is None:
            raise NotImplementedError
        return retval

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


def _MPIOperator_get_local_spaces(self, source, pickle_local_spaces):
    self = mpi.get_object(self)
    local_space = self.source if source else self.range
    if not pickle_local_spaces:
        local_space = _register_local_space(local_space)
    local_spaces = mpi.comm.gather(local_space, root=0)
    if mpi.rank0:
        return tuple(local_spaces)


def _MPIOperator_apply(self, U, U_ind, mu):
    return self.apply(_indexed(U, U_ind), mu=mu)


def _MPIOperator_apply2(self, V, V_ind, U, U_ind, mu):
    return self.apply2(_indexed(V, V_ind), _indexed(U, U_ind), mu=mu)


def _MPIOperator_pairwise_apply2(self, V, V_ind, U, U_ind, mu):
    return self.pairwise_apply2(_indexed(V, V_ind), _indexed(U, U_ind), mu=mu)


def _MPIOperator_apply_adjoint(self, V, V_ind, mu):
    return self.apply_adjoint(_indexed(V, V_ind), mu=mu)


def _MPIOperator_apply_inverse(self, V, V_ind, mu, initial_guess, least_squares):
    return self.apply_inverse(_indexed(V, V_ind), mu=mu,
                              initial_guess=initial_guess, least_squares=least_squares)


def _MPIOperator_apply_inverse_adjoint(self, U, U_ind, mu, initial_guess, least_squares):
    return self.apply_inverse_ajdoint(_indexed(U, U_ind), mu=mu,
                                      initial_guess=initial_guess, least_squares=least_squares)


def _MPIOperator_jacobian(self, U, U_ind, mu):
    return self.jacobian(_indexed(U, U_ind), mu=mu)


def _MPIOperator_assemble_lincomb(operators, coefficients, identity_shift, name):
    operators = [mpi.get_object(op) for op in operators]
    return mpi.manage_object(operators[0]._assemble_lincomb(operators, coefficients, identity_shift, name=name))


def _MPIOperator_restriced(self, dofs):
    try:
        return self.restricted(dofs)
    except NotImplementedError:
        return None


def _MPIOperator_assemble(self, mu):
    self = mpi.get_object(self)
    assembled_op = self.assemble(mu)
    if assembled_op is not self:
        return mpi.manage_object(assembled_op)
    else:
        return None


def mpi_wrap_operator(obj_id, mpi_range, mpi_source, with_apply2=False, pickle_local_spaces=True,
                      space_type=MPIVectorSpace):
    """Wrap MPI distributed local |Operators| to a global |Operator| on rank 0.

    Given MPI distributed local |Operators| referred to by the
    :class:`~pymor.tools.mpi.ObjectId` `obj_id`, return a new |Operator|
    which manages these distributed operators from rank 0. This
    is done by instantiating :class:`MPIOperator`. Additionally, the
    structure of the wrapped operators is preserved. E.g. |LincombOperators|
    will be wrapped as a |LincombOperator| of :class:`MPIOperators <MPIOperator>`.

    Parameters
    ----------
    See :class:`MPIOperator`.

    Returns
    -------
    The wrapped |Operator|.
    """
    op = mpi.get_object(obj_id)
    if isinstance(op, LincombOperator):
        obj_ids = mpi.call(_mpi_wrap_operator_LincombOperator_manage_operators, obj_id)
        return LincombOperator([mpi_wrap_operator(o, mpi_range, mpi_source, with_apply2, pickle_local_spaces,
                                                  space_type)
                                for o in obj_ids], op.coefficients, name=op.name)
    elif isinstance(op, VectorArrayOperator):
        array_obj_id, local_spaces = mpi.call(_mpi_wrap_operator_VectorArrayOperator_manage_array,
                                              obj_id, pickle_local_spaces)
        if all(ls == local_spaces[0] for ls in local_spaces):
            local_spaces = (local_spaces[0],)
        return VectorArrayOperator(space_type(local_spaces).make_array(array_obj_id),
                                   adjoint=op.adjoint, name=op.name)
    else:
        return MPIOperator(obj_id, mpi_range, mpi_source, with_apply2, pickle_local_spaces, space_type)


def _mpi_wrap_operator_LincombOperator_manage_operators(obj_id):
    op = mpi.get_object(obj_id)
    obj_ids = [mpi.manage_object(o) for o in op.operators]
    mpi.remove_object(obj_id)
    if mpi.rank0:
        return obj_ids


def _mpi_wrap_operator_VectorArrayOperator_manage_array(obj_id, pickle_local_spaces):
    op = mpi.get_object(obj_id)
    array_obj_id = mpi.manage_object(op.array)
    local_space = op.array.space
    if not pickle_local_spaces:
        local_space = _register_local_space(local_space)
    local_spaces = mpi.comm.gather(local_space, root=0)
    mpi.remove_object(obj_id)
    if mpi.rank0:
        return array_obj_id, tuple(local_spaces)

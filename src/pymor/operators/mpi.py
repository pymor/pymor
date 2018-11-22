# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.tools import mpi
from pymor.vectorarrays.mpi import MPIVectorSpace, _register_local_space
from pymor.vectorarrays.numpy import NumpyVectorSpace


class MPIOperator(OperatorBase):
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
    with_apply2
        Set to `True` if the operator implementation has its own
        MPI aware implementation of `apply2` and `pairwise_apply2`.
        Otherwise, the default implementations using `apply` and
        :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.dot`
        will be used.
    pickle_local_spaces
        If `pickle_local_spaces` is `False`, a unique identifier
        is computed for each local source/range |VectorSpace|, which is then
        transferred to rank 0 instead of the true |VectorSpace|. This
        allows the useage of :class:`~pymor.vectorarrays.mpi.MPIVectorArray`
        even when the local |VectorSpaces| are not picklable.
    space_type
        This class will be used to wrap the local |VectorArrays|
        returned by the local operators into an MPI distributed
        |VectorArray| managed from rank 0. By default,
        :class:`~pymor.vectorarrays.mpi.MPIVectorSpace` will be used,
        other options are :class:`~pymor.vectorarrays.mpi.MPIVectorSpaceAutoComm`
        and :class:`~pymor.vectorarrays.mpi.MPIVectorSpaceNoComm`.
    """

    def __init__(self, obj_id, with_apply2=False, pickle_local_spaces=True, space_type=MPIVectorSpace):
        self.obj_id = obj_id
        self.op = op = mpi.get_object(obj_id)
        assert not op.source.id == op.range.id == None
        self.with_apply2 = with_apply2
        self.pickle_local_spaces = pickle_local_spaces
        self.space_type = space_type
        self.linear = op.linear
        self.name = op.name
        self.build_parameter_type(op)
        if op.source.id is None:
            self.source = op.source
        else:
            local_spaces = mpi.call(_MPIOperator_get_local_spaces, obj_id, True, pickle_local_spaces)
            if all(ls == local_spaces[0] for ls in local_spaces):
                local_spaces = (local_spaces[0],)
            self.source = space_type(local_spaces)
        if op.range.id is None:
            self.range = op.range
        else:
            local_spaces = mpi.call(_MPIOperator_get_local_spaces, obj_id, False, pickle_local_spaces)
            if all(ls == local_spaces[0] for ls in local_spaces):
                local_spaces = (local_spaces[0],)
            self.range = space_type(local_spaces)

    def apply(self, U, mu=None):
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.source.id is None else U.obj_id
        if self.range.id is None:
            return mpi.call(mpi.method_call, self.obj_id, 'apply', U, mu=mu)
        else:
            return self.range.make_array(mpi.call(mpi.method_call_manage, self.obj_id, 'apply', U, mu=mu))

    def as_range_array(self, mu=None):
        mu = self.parse_parameter(mu)
        return self.range.make_array(mpi.call(mpi.method_call_manage, self.obj_id, 'as_range_array', mu=mu))

    def as_source_array(self, mu=None):
        mu = self.parse_parameter(mu)
        return self.source.make_array(mpi.call(mpi.method_call_manage, self.obj_id, 'as_source_array', mu=mu))

    def apply2(self, V, U, mu=None):
        if not self.with_apply2:
            return super().apply2(V, U, mu=mu)
        assert V in self.range
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.source.id is None else U.obj_id
        V = V if self.range.id is None else V.obj_id
        return mpi.call(mpi.method_call, self.obj_id, 'apply2', V, U, mu=mu)

    def pairwise_apply2(self, V, U, mu=None):
        if not self.with_apply2:
            return super().pairwise_apply2(V, U, mu=mu)
        assert V in self.range
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.source.id is None else U.obj_id
        V = V if self.range.id is None else V.obj_id
        return mpi.call(mpi.method_call, self.obj_id, 'pairwise_apply2', V, U, mu=mu)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        mu = self.parse_parameter(mu)
        V = V if self.range.id is None else V.obj_id
        if self.source.id is None:
            return mpi.call(mpi.method_call, self.obj_id, 'apply_adjoint', V, mu=mu)
        else:
            return self.source.make_array(
                mpi.call(mpi.method_call_manage, self.obj_id, 'apply_adjoint', V, mu=mu)
            )

    def apply_inverse(self, V, mu=None, least_squares=False):
        if self.source.id is None or self.range.id is None:
            raise NotImplementedError
        assert V in self.range
        mu = self.parse_parameter(mu)
        return self.source.make_array(mpi.call(mpi.method_call_manage, self.obj_id, 'apply_inverse',
                                               V.obj_id, mu=mu, least_squares=least_squares))

    def jacobian(self, U, mu=None):
        assert U in self.source
        mu = self.parse_parameter(mu)
        return self.with_(obj_id=mpi.call(mpi.method_call_manage, self.obj_id, 'jacobian', U.obj_id, mu=mu))

    def assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        return self.with_(obj_id=mpi.call(mpi.method_call_manage, self.obj_id, 'assemble', mu=mu))

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
        if not all(isinstance(op, MPIOperator) for op in operators):
            return None
        assert solver_options is None
        operators = [op.obj_id for op in operators]
        obj_id = mpi.call(_MPIOperator_assemble_lincomb, operators, coefficients, name=name)
        op = mpi.get_object(obj_id)
        if op is None:
            mpi.call(mpi.remove_object, obj_id)
            return None
        else:
            return self.with_(obj_id=obj_id)

    def restricted(self, dofs):
        return mpi.call(mpi.method_call, self.obj_id, 'restricted', dofs)

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


def _MPIOperator_assemble_lincomb(operators, coefficients, name):
    operators = [mpi.get_object(op) for op in operators]
    return mpi.manage_object(operators[0].assemble_lincomb(operators, coefficients, name=name))


def mpi_wrap_operator(obj_id, with_apply2=False, pickle_local_spaces=True, space_type=MPIVectorSpace):
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
        return LincombOperator([mpi_wrap_operator(o, with_apply2, pickle_local_spaces, space_type)
                                for o in obj_ids], op.coefficients, name=op.name)
    elif isinstance(op, VectorArrayOperator):
        array_obj_id, local_spaces = mpi.call(_mpi_wrap_operator_VectorArrayOperator_manage_array,
                                              obj_id, pickle_local_spaces)
        if all(ls == local_spaces[0] for ls in local_spaces):
            local_spaces = (local_spaces[0],)
        return VectorArrayOperator(space_type(local_spaces).make_array(array_obj_id),
                                   adjoint=op.adjoint, name=op.name)
    else:
        return MPIOperator(obj_id, with_apply2, pickle_local_spaces, space_type)


def _mpi_wrap_operator_LincombOperator_manage_operators(obj_id):
    op = mpi.get_object(obj_id)
    obj_ids = [mpi.manage_object(o) for o in op.operators]
    mpi.remove_object(obj_id)
    if mpi.rank0:
        return obj_ids


def _mpi_wrap_operator_VectorArrayOperator_manage_array(obj_id, pickle_local_spaces):
    op = mpi.get_object(obj_id)
    array_obj_id = mpi.manage_object(op._array)
    local_space = op._array.space
    if not pickle_local_spaces:
        local_space = _register_local_space(local_space)
    local_spaces = mpi.comm.gather(local_space, root=0)
    mpi.remove_object(obj_id)
    if mpi.rank0:
        return array_obj_id, tuple(local_spaces)

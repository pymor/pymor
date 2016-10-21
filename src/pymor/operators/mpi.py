# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.tools import mpi
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.mpi import MPIVectorArray, _register_subtype
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
    pickle_subtypes
        If `pickle_subtypes` is `False`, a unique identifier
        is computed for each local source/range subtype, which is then
        transferred to rank 0 instead of the true subtype. This
        allows the useage of :class:`~pymor.vectorarrays.mpi.MPIVectorArray`
        even when the local subtypes are not picklable.
    array_type
        This class will be used to wrap the local |VectorArrays|
        returned by the local operators into an MPI distributed
        |VectorArray| managed from rank 0. By default,
        :class:`~pymor.vectorarrays.mpi.MPIVectorArray` will be used,
        other options are :class:`~pymor.vectorarrays.mpi.MPIVectorArrayAutoComm`
        and :class:`~pymor.vectorarrays.mpi.MPIVectorArrayNoComm`.
    """

    def __init__(self, obj_id, with_apply2=False,
                 pickle_subtypes=True, array_type=MPIVectorArray):
        self.obj_id = obj_id
        self.op = op = mpi.get_object(obj_id)
        self.with_apply2 = with_apply2
        self.pickle_subtypes = pickle_subtypes
        self.array_type = array_type
        self.linear = op.linear
        self.name = op.name
        self.build_parameter_type(op)
        if op.is_vector or op.is_function:
            self.source = NumpyVectorSpace(1)
            assert self.source == op.source
        else:
            subtypes = mpi.call(_MPIOperator_get_source_subtypes, obj_id, pickle_subtypes)
            if all(subtype == subtypes[0] for subtype in subtypes):
                subtypes = (subtypes[0],)
            self.source = VectorSpace(array_type, (op.source.type, subtypes))
        if op.is_functional or op.is_function:
            self.range = NumpyVectorSpace(1)
            assert self.range == op.range
        else:
            subtypes = mpi.call(_MPIOperator_get_range_subtypes, obj_id, pickle_subtypes)
            if all(subtype == subtypes[0] for subtype in subtypes):
                subtypes = (subtypes[0],)
            self.range = VectorSpace(array_type, (op.range.type, subtypes))
        self.kind = op.kind

    def apply(self, U, mu=None):
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.is_vector else U.obj_id
        if self.is_functional:
            return mpi.call(mpi.method_call, self.obj_id, 'apply', U, mu=mu)
        else:
            space = self.range
            return space.type(space.subtype[0], space.subtype[1],
                              mpi.call(mpi.method_call_manage, self.obj_id, 'apply', U, mu=mu))

    def as_vector(self, mu=None):
        mu = self.parse_parameter(mu)
        if self.is_functional:
            space = self.source
            return space.type(space.subtype[0], space.subtype[1],
                              mpi.call(mpi.method_call_manage, self.obj_id, 'as_vector', mu=mu))
        else:
            raise NotImplementedError

    def apply2(self, V, U, mu=None):
        if not self.with_apply2:
            return super().apply2(V, U, mu=mu)
        assert V in self.range
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.is_vector else U.obj_id
        V = V if self.is_functional else V.obj_id
        return mpi.call(mpi.method_call, self.obj_id, 'apply2',
                        V, U, mu=mu)

    def pairwise_apply2(self, V, U, mu=None):
        if not self.with_apply2:
            return super().pairwise_apply2(V, U, mu=mu)
        assert V in self.range
        assert U in self.source
        mu = self.parse_parameter(mu)
        U = U if self.is_vector else U.obj_id
        V = V if self.is_functional else V.obj_id
        return mpi.call(mpi.method_call, self.obj_id, 'pairwise_apply2',
                        V, U, mu=mu)

    def apply_adjoint(self, U, mu=None, source_product=None, range_product=None):
        assert U in self.range
        mu = self.parse_parameter(mu)
        U = U if self.is_functional else U.obj_id
        source_product = source_product and source_product.obj_id
        range_product = range_product and range_product.obj_id
        if self.is_vector:
            return mpi.call(mpi.method_call, self.obj_id, 'apply_adjoint',
                            U, mu=mu, source_product=source_product, range_product=range_product)
        else:
            space = self.source
            return space.type(space.subtype[0], space.subtype[1],
                              mpi.call(mpi.method_call_manage, self.obj_id, 'apply_adjoint',
                                       U, mu=mu, source_product=source_product, range_product=range_product))

    def apply_inverse(self, V, mu=None, least_squares=False):
        if not self.is_operator:
            raise NotImplementedError
        assert V in self.range
        mu = self.parse_parameter(mu)
        space = self.source
        return space.type(space.subtype[0], space.subtype[1],
                          mpi.call(mpi.method_call_manage, self.obj_id, 'apply_inverse',
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


def _MPIOperator_get_source_subtypes(self, pickle_subtypes):
    self = mpi.get_object(self)
    subtype = self.source.subtype
    if not pickle_subtypes:
        subtype = _register_subtype(subtype)
    subtypes = mpi.comm.gather(subtype, root=0)
    if mpi.rank0:
        return tuple(subtypes)


def _MPIOperator_get_range_subtypes(self, pickle_subtypes):
    self = mpi.get_object(self)
    subtype = self.range.subtype
    if not pickle_subtypes:
        subtype = _register_subtype(subtype)
    subtypes = mpi.comm.gather(subtype, root=0)
    if mpi.rank0:
        return tuple(subtypes)


def _MPIOperator_assemble_lincomb(operators, coefficients, name):
    operators = [mpi.get_object(op) for op in operators]
    return mpi.manage_object(operators[0].assemble_lincomb(operators, coefficients, name=name))


def mpi_wrap_operator(obj_id, with_apply2=False,
                      pickle_subtypes=True, array_type=MPIVectorArray):
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
        return LincombOperator([mpi_wrap_operator(o, with_apply2, pickle_subtypes, array_type)
                                for o in obj_ids], op.coefficients, name=op.name)
    elif isinstance(op, VectorArrayOperator):
        array_obj_id, subtypes = mpi.call(_mpi_wrap_operator_VectorArrayOperator_manage_array, obj_id, pickle_subtypes)
        if all(subtype == subtypes[0] for subtype in subtypes):
            subtypes = (subtypes[0],)
        return VectorArrayOperator(array_type(type(op._array), subtypes, array_obj_id),
                                   transposed=op.transposed, name=op.name)
    else:
        return MPIOperator(obj_id, with_apply2, pickle_subtypes, array_type)


def _mpi_wrap_operator_LincombOperator_manage_operators(obj_id):
    op = mpi.get_object(obj_id)
    obj_ids = [mpi.manage_object(o) for o in op.operators]
    mpi.remove_object(obj_id)
    if mpi.rank0:
        return obj_ids


def _mpi_wrap_operator_VectorArrayOperator_manage_array(obj_id, pickle_subtypes):
    op = mpi.get_object(obj_id)
    array_obj_id = mpi.manage_object(op._array)
    subtype = op._array.subtype
    if not pickle_subtypes:
        subtype = _register_subtype(subtype)
    subtypes = mpi.comm.gather(subtype, root=0)
    mpi.remove_object(obj_id)
    if mpi.rank0:
        return array_obj_id, tuple(subtypes)

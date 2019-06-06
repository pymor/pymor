# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from collections import namedtuple

from pymor.core.interfaces import ImmutableInterface
from pymor.models.interfaces import ModelInterface
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.mpi import mpi_wrap_operator
from pymor.tools import mpi
from pymor.vectorarrays.mpi import MPIVectorSpace


class MPIModel:
    """Wrapper class mixin for MPI distributed |Models|.

    Given a single-rank implementation of a |Model|, this
    wrapper class uses the event loop from :mod:`pymor.tools.mpi`
    to allow an MPI distributed usage of the |Model|.
    The underlying implementation needs to be MPI aware.
    In particular, the model's
    :meth:`~pymor.models.interfaces.ModelInterface.solve`
    method has to perform an MPI parallel solve of the model.

    Note that this class is not intended to be instantiated directly.
    Instead, you should use :func:`mpi_wrap_model`.
    """

    def __init__(self, obj_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_id = obj_id
        self.visualizer = MPIVisualizer(obj_id)

    def _solve(self, mu=None):
        return self.solution_space.make_array(
            mpi.call(mpi.method_call_manage, self.obj_id, "solve", mu=mu)
        )

    def visualize(self, U, **kwargs):
        self.visualizer.visualize(U, self, **kwargs)

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


class MPIVisualizer(ImmutableInterface):
    def __init__(self, m_obj_id):
        self.m_obj_id = m_obj_id

    def visualize(self, U, m, **kwargs):
        if isinstance(U, tuple):
            U = tuple(u.obj_id for u in U)
        else:
            U = U.obj_id
        mpi.call(_MPIVisualizer_visualize, self.m_obj_id, U, **kwargs)


def _MPIVisualizer_visualize(m, U, **kwargs):
    m = mpi.get_object(m)
    if isinstance(U, tuple):
        U = tuple(mpi.get_object(u) for u in U)
    else:
        U = mpi.get_object(U)
    m.visualize(U, **kwargs)


def mpi_wrap_model(
    local_models,
    mpi_spaces=("STATE",),
    use_with=True,
    with_apply2=False,
    pickle_local_spaces=True,
    space_type=MPIVectorSpace,
    base_type=None,
):
    """Wrap MPI distributed local |Models| to a global |Model| on rank 0.

    Given MPI distributed local |Models| referred to by the
    :class:`~pymor.tools.mpi.ObjectId` `local_models`, return a new |Model|
    which manages these distributed models from rank 0. This
    is done by first wrapping all |Operators| of the |Model| using
    :func:`~pymor.operators.mpi.mpi_wrap_operator`.

    Alternatively, `local_models` can be a callable (with no arguments)
    which is then called on each rank to instantiate the local |Models|.

    When `use_with` is `False`, an :class:`MPIModel` is instantiated
    with the wrapped operators. A call to
    :meth:`~pymor.models.interfaces.ModelInterface.solve`
    will then use an MPI parallel call to the
    :meth:`~pymor.models.interfaces.ModelInterface.solve`
    methods of the wrapped local |Models| to obtain the solution.
    This is usually what you want when the actual solve is performed by
    an implementation in the external solver.

    When `use_with` is `True`, :meth:`~pymor.core.interfaces.ImmutableInterface.with_`
    is called on the local |Model| on rank 0, to obtain a new
    |Model| with the wrapped MPI |Operators|. This is mainly useful
    when the local models are generic |Models| as in
    :mod:`pymor.models.basic` and
    :meth:`~pymor.models.interfaces.ModelInterface.solve`
    is implemented directly in pyMOR via operations on the contained
    |Operators|.

    Parameters
    ----------
    local_models
        :class:`~pymor.tools.mpi.ObjectId` of the local |Models|
        on each rank or a callable generating the |Models|.
    mpi_spaces
        List of types or ids of |VectorSpaces| which are MPI distributed
        and need to be wrapped.
    use_with
        See above.
    with_apply2
        See :class:`~pymor.operators.mpi.MPIOperator`.
    pickle_local_spaces
        See :class:`~pymor.operators.mpi.MPIOperator`.
    space_type
        See :class:`~pymor.operators.mpi.MPIOperator`.
    """

    assert use_with or isinstance(base_type, ModelInterface)

    if not isinstance(local_models, mpi.ObjectId):
        local_models = mpi.call(mpi.function_call_manage, local_models)

    attributes = mpi.call(
        _mpi_wrap_model_manage_operators, local_models, mpi_spaces, use_with, base_type
    )

    wrapped_attributes = {
        k: _map_children(
            lambda v: mpi_wrap_operator(
                *v,
                with_apply2=with_apply2,
                pickle_local_spaces=pickle_local_spaces,
                space_type=space_type
            )
            if isinstance(v, _OperatorToWrap)
            else v,
            v,
        )
        for k, v in attributes.items()
    }

    if use_with:
        m = mpi.get_object(local_models)
        if m.visualizer:
            wrapped_attributes["visualizer"] = MPIVisualizer(local_models)
        return m.with_(cache_region=None, **wrapped_attributes)
    else:

        class MPIWrappedModel(MPIModel, base_type):
            pass

        return MPIWrappedModel(local_models, **wrapped_attributes)


_OperatorToWrap = namedtuple("_OperatorToWrap", "operator mpi_range mpi_source")


def _mpi_wrap_model_manage_operators(obj_id, mpi_spaces, use_with, base_type):
    m = mpi.get_object(obj_id)

    attributes_to_consider = (
        m._init_arguments if use_with else base_type._init_arguments
    )
    attributes = {k: getattr(m, k) for k in attributes_to_consider}

    def process_attribute(v):
        if isinstance(v, OperatorInterface):
            mpi_range = type(v.range) in mpi_spaces or v.range.id in mpi_spaces
            mpi_source = type(v.source) in mpi_spaces or v.source.id in mpi_spaces
            if mpi_range or mpi_source:
                return _OperatorToWrap(mpi.manage_object(v), mpi_range, mpi_source)
            else:
                return v
        else:
            return v

    managed_attributes = {
        k: _map_children(process_attribute, v)
        for k, v in sorted(attributes.items())
        if k not in {"cache_region", "visualizer"}
    }
    if mpi.rank0:
        return managed_attributes


def _map_children(f, obj):
    if isinstance(obj, dict):
        return {k: f(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple, set)) and not isinstance(obj, _OperatorToWrap):
        return type(obj)(f(v) for v in obj)
    else:
        return f(obj)
